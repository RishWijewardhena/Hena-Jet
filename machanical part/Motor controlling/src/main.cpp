#include <Arduino.h>

#include <cctype>
#include <cmath>

#include <cstdlib>
#include <cstring>

#include "motion_math.h"

namespace {

constexpr double MOTOR_RPM = 0.25;
constexpr double START_RPM = 0.25;
constexpr double ACCELERATION_RPM_PER_SECOND = 3.0;
constexpr double MINIMUM_CONTINUOUS_RPM = 0.05;
constexpr double MAXIMUM_CONTINUOUS_RPM = 2.0;
constexpr uint32_t SEGMENT_DELAY_MS = 200;
constexpr uint32_t PULSE_HIGH_US = 100;
constexpr uint32_t MINIMUM_PULSES_PER_REVOLUTION = 1;
constexpr uint32_t MAXIMUM_PULSES_PER_REVOLUTION = 100000;
constexpr size_t COMMAND_BUFFER_SIZE = 64;

constexpr uint8_t DIRECTION_PIN = D6;
constexpr uint8_t ENABLE_PIN = D5;
constexpr uint8_t ALARM_PIN = D4;
constexpr uint8_t PULSE_PIN = D3;

// D4 must be driven by a 0-3.3 V conditioned signal. Never connect a 5 V
// alarm signal directly to the XIAO ESP32-S3 GPIO.
constexpr uint8_t ALARM_ACTIVE_LEVEL = HIGH;

enum class MotionState {
    Idle,
    Stepping,
    WaitingBetweenSegments,
    AwaitingNextSegment,
};

MotionState motionState = MotionState::Idle;
uint32_t activePulsesPerRevolution = 0;
double requestedIncrementDegrees = 0.0;
double completedDegrees = 0.0;
double currentSegmentDegrees = 0.0;
uint32_t emittedPulses = 0;
uint32_t segmentPulseTotal = 0;
uint32_t segmentPulsesEmitted = 0;
uint32_t nextPulseTransitionUs = 0;
uint32_t currentPulsePeriodUs = 0;
uint32_t segmentWaitStartedMs = 0;
double currentPulseRateHz = 0.0;
double startPulseRateHz = 0.0;
double maximumPulseRateHz = 0.0;
double pulseAccelerationHzPerSecond = 0.0;
double stopInitialRateHz = 0.0;
uint32_t stopBrakingPulseTotal = 0;
uint32_t stopBrakingPulsesEmitted = 0;
bool pulseIsHigh = false;
bool stopRequested = false;
bool currentPulseIsBraking = false;
bool synchronizedSequence = false;
bool continuousSequence = false;
double nextCaptureAngleDegrees = 0.0;

char commandBuffer[COMMAND_BUFFER_SIZE] = {};
size_t commandLength = 0;
bool commandOverflow = false;

volatile uint32_t alarmEventsPending = 0;

void IRAM_ATTR onAlarmRisingEdge()
{
    if (alarmEventsPending < UINT32_MAX) {
        ++alarmEventsPending;
    }
}

bool isMotionActive()
{
    return motionState != MotionState::Idle;
}

void printDegrees(double degrees)
{
    char text[24];
    snprintf(text, sizeof(text), "%.6f", degrees);

    char *end = text + strlen(text) - 1;
    while (end > text && *end == '0') {
        *end-- = '\0';
    }
    if (end > text && *end == '.') {
        *end = '\0';
    }

    Serial.print(text);
}

void stopMotion(bool announce)
{
    digitalWrite(PULSE_PIN, LOW);
    pulseIsHigh = false;
    motionState = MotionState::Idle;
    segmentPulseTotal = 0;
    segmentPulsesEmitted = 0;
    currentPulseRateHz = 0.0;
    currentPulsePeriodUs = 0;
    stopInitialRateHz = 0.0;
    stopBrakingPulseTotal = 0;
    stopBrakingPulsesEmitted = 0;
    stopRequested = false;
    currentPulseIsBraking = false;
    continuousSequence = false;
    nextCaptureAngleDegrees = 0.0;

    if (announce) {
        Serial.println("stopped");
    }
}

void finishCurrentSegment();

void reportContinuousCaptureEvents()
{
    while (nextCaptureAngleDegrees > 0.0
           && nextCaptureAngleDegrees <= motion::FULL_REVOLUTION_DEGREES) {
        const uint32_t targetPulses = motion::continuousPulseTarget(
            nextCaptureAngleDegrees, activePulsesPerRevolution);
        if (emittedPulses < targetPulses) {
            return;
        }

        Serial.print("angle_ok,");
        printDegrees(nextCaptureAngleDegrees);
        Serial.print(",");
        Serial.println(emittedPulses);

        if (nextCaptureAngleDegrees + 1e-9
            >= motion::FULL_REVOLUTION_DEGREES) {
            nextCaptureAngleDegrees = 0.0;
        } else {
            nextCaptureAngleDegrees = std::fmin(
                nextCaptureAngleDegrees + requestedIncrementDegrees,
                motion::FULL_REVOLUTION_DEGREES);
        }
    }
}

void beginNextSegment()
{
    currentSegmentDegrees =
        motion::nextSegmentDegrees(completedDegrees, requestedIncrementDegrees);
    const double targetDegrees = completedDegrees + currentSegmentDegrees;
    const uint32_t targetPulses =
        motion::cumulativePulseTarget(targetDegrees, activePulsesPerRevolution);

    segmentPulseTotal =
        motion::additionalPulsesToTarget(emittedPulses, targetPulses);
    if (segmentPulseTotal == 0) {
        finishCurrentSegment();
        return;
    }

    segmentPulsesEmitted = 0;
    currentPulseRateHz = 0.0;
    currentPulsePeriodUs = 0;
    currentPulseIsBraking = false;
    stopRequested = false;
    pulseIsHigh = false;
    digitalWrite(PULSE_PIN, LOW);
    nextPulseTransitionUs = micros();
    motionState = MotionState::Stepping;
}

void finishCurrentSegment()
{
    completedDegrees += currentSegmentDegrees;
    if (completedDegrees > motion::FULL_REVOLUTION_DEGREES) {
        completedDegrees = motion::FULL_REVOLUTION_DEGREES;
    }

    printDegrees(currentSegmentDegrees);
    Serial.println(" degree ok");

    if (completedDegrees + 1e-9 >= motion::FULL_REVOLUTION_DEGREES) {
        motionState = MotionState::Idle;
        Serial.println("completed");
        return;
    }

    if (synchronizedSequence) {
        motionState = MotionState::AwaitingNextSegment;
    } else {
        segmentWaitStartedMs = millis();
        motionState = MotionState::WaitingBetweenSegments;
    }
}

void serviceMotion()
{
    if (motionState == MotionState::WaitingBetweenSegments) {
        if (static_cast<uint32_t>(millis() - segmentWaitStartedMs)
            >= SEGMENT_DELAY_MS) {
            beginNextSegment();
        }
        return;
    }

    if (motionState != MotionState::Stepping) {
        return;
    }

    const uint32_t now = micros();
    if (static_cast<int32_t>(now - nextPulseTransitionUs) < 0) {
        return;
    }

    if (!pulseIsHigh) {
        if (stopRequested) {
            if (stopBrakingPulseTotal == 0) {
                stopMotion(true);
                return;
            }
            currentPulseRateHz = motion::sCurveStopRateHz(
                stopBrakingPulsesEmitted,
                stopBrakingPulseTotal,
                stopInitialRateHz,
                startPulseRateHz);
            currentPulseIsBraking = true;
        } else {
            currentPulseRateHz = motion::sCurvePulseRateHz(
                segmentPulsesEmitted,
                segmentPulseTotal,
                startPulseRateHz,
                maximumPulseRateHz,
                pulseAccelerationHzPerSecond);
            currentPulseIsBraking = false;
        }

        const uint32_t halfPeriodUs =
            motion::pulseHalfPeriodUs(currentPulseRateHz);
        currentPulsePeriodUs = halfPeriodUs * 2;
        if (currentPulsePeriodUs <= PULSE_HIGH_US) {
            Serial.println("error,invalid_speed_config");
            stopMotion(false);
            return;
        }

        digitalWrite(PULSE_PIN, HIGH);
        pulseIsHigh = true;
        nextPulseTransitionUs = now + PULSE_HIGH_US;
        return;
    }

    digitalWrite(PULSE_PIN, LOW);
    pulseIsHigh = false;
    ++emittedPulses;
    if (currentPulseIsBraking) {
        ++stopBrakingPulsesEmitted;
    } else {
        ++segmentPulsesEmitted;
    }

    if (stopRequested) {
        if (stopBrakingPulsesEmitted >= stopBrakingPulseTotal) {
            stopMotion(true);
        } else {
            nextPulseTransitionUs =
                now + currentPulsePeriodUs - PULSE_HIGH_US;
        }
        return;
    }

    if (continuousSequence) {
        reportContinuousCaptureEvents();
        if (segmentPulsesEmitted >= segmentPulseTotal) {
            stopMotion(false);
            Serial.println("completed");
        } else {
            nextPulseTransitionUs =
                now + currentPulsePeriodUs - PULSE_HIGH_US;
        }
        return;
    }

    if (segmentPulsesEmitted == segmentPulseTotal) {
        finishCurrentSegment();
    } else {
        nextPulseTransitionUs = now + currentPulsePeriodUs - PULSE_HIGH_US;
    }
}

void requestControlledStop()
{
    if (motionState == MotionState::Idle
        || motionState == MotionState::WaitingBetweenSegments) {
        stopMotion(true);
        return;
    }

    stopRequested = true;
    stopInitialRateHz = currentPulseRateHz;
    stopBrakingPulseTotal = motion::brakingPulseCount(
        currentPulseRateHz,
        startPulseRateHz,
        pulseAccelerationHzPerSecond);
    stopBrakingPulsesEmitted = 0;
    if (!pulseIsHigh && stopBrakingPulseTotal == 0) {
        stopMotion(true);
    }
}

char *trimWhitespace(char *text)
{
    while (*text != '\0' && isspace(static_cast<unsigned char>(*text))) {
        ++text;
    }

    char *end = text + strlen(text);
    while (end > text && isspace(static_cast<unsigned char>(end[-1]))) {
        --end;
    }
    *end = '\0';
    return text;
}

void lowercase(char *text)
{
    for (; *text != '\0'; ++text) {
        *text = static_cast<char>(tolower(static_cast<unsigned char>(*text)));
    }
}

void startContinuousSequence(
    double incrementDegrees,
    uint32_t pulsesPerRevolution,
    double targetRpm)
{
    if (isMotionActive()) {
        Serial.println("error,busy");
        return;
    }
    if (digitalRead(ALARM_PIN) == ALARM_ACTIVE_LEVEL) {
        Serial.println("error,alarm_active");
        return;
    }
    if (pulsesPerRevolution < MINIMUM_PULSES_PER_REVOLUTION
        || pulsesPerRevolution > MAXIMUM_PULSES_PER_REVOLUTION) {
        Serial.println("error,invalid_ppr");
        return;
    }
    if (!motion::isValidIncrement(incrementDegrees, pulsesPerRevolution)) {
        Serial.println("error,invalid_degree");
        return;
    }
    if (!std::isfinite(targetRpm) || targetRpm < MINIMUM_CONTINUOUS_RPM
        || targetRpm > MAXIMUM_CONTINUOUS_RPM) {
        Serial.println("error,invalid_rpm");
        return;
    }

    activePulsesPerRevolution = pulsesPerRevolution;
    startPulseRateHz = std::fmin(START_RPM, targetRpm)
        * activePulsesPerRevolution / 60.0;
    maximumPulseRateHz = targetRpm * activePulsesPerRevolution / 60.0;
    pulseAccelerationHzPerSecond =
        ACCELERATION_RPM_PER_SECOND * activePulsesPerRevolution / 60.0;
    requestedIncrementDegrees = incrementDegrees;
    completedDegrees = 0.0;
    currentSegmentDegrees = motion::continuousRunoutDegrees(incrementDegrees);
    emittedPulses = 0;
    segmentPulsesEmitted = 0;
    segmentPulseTotal = motion::continuousPulseTarget(
        currentSegmentDegrees, activePulsesPerRevolution);
    stopRequested = false;
    synchronizedSequence = false;
    continuousSequence = true;
    nextCaptureAngleDegrees = std::fmin(
        incrementDegrees, motion::FULL_REVOLUTION_DEGREES);
    currentPulseRateHz = 0.0;
    currentPulsePeriodUs = 0;
    currentPulseIsBraking = false;
    pulseIsHigh = false;
    digitalWrite(PULSE_PIN, LOW);
    nextPulseTransitionUs = micros();
    motionState = MotionState::Stepping;

    Serial.print("started_continuous,");
    printDegrees(incrementDegrees);
    Serial.print(",");
    Serial.print(activePulsesPerRevolution);
    Serial.print(",");
    printDegrees(targetRpm);
    Serial.println();
}

void startSequence(
    double incrementDegrees,
    uint32_t pulsesPerRevolution,
    bool synchronized)
{
    if (isMotionActive()) {
        Serial.println("error,busy");
        return;
    }
    if (digitalRead(ALARM_PIN) == ALARM_ACTIVE_LEVEL) {
        Serial.println("error,alarm_active");
        return;
    }
    if (pulsesPerRevolution < MINIMUM_PULSES_PER_REVOLUTION
        || pulsesPerRevolution > MAXIMUM_PULSES_PER_REVOLUTION) {
        Serial.println("error,invalid_ppr");
        return;
    }
    if (!motion::isValidIncrement(incrementDegrees, pulsesPerRevolution)) {
        Serial.println("error,invalid_degree");
        return;
    }

    activePulsesPerRevolution = pulsesPerRevolution;
    startPulseRateHz = START_RPM * activePulsesPerRevolution / 60.0;
    maximumPulseRateHz = MOTOR_RPM * activePulsesPerRevolution / 60.0;
    pulseAccelerationHzPerSecond =
        ACCELERATION_RPM_PER_SECOND * activePulsesPerRevolution / 60.0;
    requestedIncrementDegrees = incrementDegrees;
    completedDegrees = 0.0;
    currentSegmentDegrees = 0.0;
    emittedPulses = 0;
    stopRequested = false;
    synchronizedSequence = synchronized;

    Serial.print(synchronized ? "started_sync," : "started,");
    printDegrees(requestedIncrementDegrees);
    Serial.print(',');
    Serial.print(activePulsesPerRevolution);
    Serial.println();
    beginNextSegment();
}

void processCommand(char *rawCommand)
{
    char *command = trimWhitespace(rawCommand);
    lowercase(command);

    if (strcmp(command, "stop") == 0) {
        requestControlledStop();
        return;
    }

    if (strcmp(command, "next") == 0) {
        if (motionState != MotionState::AwaitingNextSegment) {
            Serial.println("error,invalid_state");
            return;
        }
        beginNextSegment();
        return;
    }

    constexpr char START_CONTINUOUS_PREFIX[] = "start_continuous,";
    if (strncmp(command, START_CONTINUOUS_PREFIX,
                sizeof(START_CONTINUOUS_PREFIX) - 1) == 0) {
        char *angleText = trimWhitespace(
            command + sizeof(START_CONTINUOUS_PREFIX) - 1);
        char *firstSeparator = strchr(angleText, ',');
        if (firstSeparator == nullptr) {
            Serial.println("error,invalid_command");
            return;
        }
        *firstSeparator = '\0';
        char *pprText = trimWhitespace(firstSeparator + 1);
        char *secondSeparator = strchr(pprText, ',');
        if (secondSeparator == nullptr) {
            Serial.println("error,invalid_command");
            return;
        }
        *secondSeparator = '\0';
        char *rpmText = trimWhitespace(secondSeparator + 1);

        char *angleEnd = nullptr;
        const double angle = strtod(angleText, &angleEnd);
        angleEnd = trimWhitespace(angleEnd);
        if (angleText == angleEnd || *angleEnd != '\0'
            || !std::isfinite(angle)) {
            Serial.println("error,invalid_degree");
            return;
        }

        char *pprEnd = nullptr;
        const unsigned long parsedPpr = strtoul(pprText, &pprEnd, 10);
        pprEnd = trimWhitespace(pprEnd);
        if (pprText == pprEnd || *pprEnd != '\0' || *pprText == '-'
            || parsedPpr > UINT32_MAX) {
            Serial.println("error,invalid_ppr");
            return;
        }

        char *rpmEnd = nullptr;
        const double rpm = strtod(rpmText, &rpmEnd);
        rpmEnd = trimWhitespace(rpmEnd);
        if (rpmText == rpmEnd || *rpmEnd != '\0'
            || !std::isfinite(rpm)) {
            Serial.println("error,invalid_rpm");
            return;
        }

        startContinuousSequence(
            angle, static_cast<uint32_t>(parsedPpr), rpm);
        return;
    }

    constexpr char START_PREFIX[] = "start,";
    constexpr char START_SYNC_PREFIX[] = "start_sync,";
    const bool synchronized = strncmp(
        command, START_SYNC_PREFIX, sizeof(START_SYNC_PREFIX) - 1) == 0;
    const bool automatic = strncmp(
        command, START_PREFIX, sizeof(START_PREFIX) - 1) == 0;
    if (!synchronized && !automatic) {
        Serial.println("error,invalid_command");
        return;
    }

    char *angleText = trimWhitespace(
        command + (synchronized
            ? sizeof(START_SYNC_PREFIX) - 1
            : sizeof(START_PREFIX) - 1));
    char *separator = strchr(angleText, ',');
    if (separator == nullptr) {
        Serial.println("error,invalid_command");
        return;
    }
    *separator = '\0';
    char *pprText = trimWhitespace(separator + 1);

    char *angleEnd = nullptr;
    const double angle = strtod(angleText, &angleEnd);
    angleEnd = trimWhitespace(angleEnd);
    if (angleText == angleEnd || *angleEnd != '\0' || !std::isfinite(angle)) {
        Serial.println("error,invalid_degree");
        return;
    }

    char *pprEnd = nullptr;
    const unsigned long parsedPpr = strtoul(pprText, &pprEnd, 10);
    pprEnd = trimWhitespace(pprEnd);
    if (pprText == pprEnd || *pprEnd != '\0' || *pprText == '-'
        || parsedPpr > UINT32_MAX) {
        Serial.println("error,invalid_ppr");
        return;
    }

    startSequence(angle, static_cast<uint32_t>(parsedPpr), synchronized);
}

void serviceSerial()
{
    while (Serial.available() > 0) {
        const char incoming = static_cast<char>(Serial.read());

        if (incoming == '\r') {
            continue;
        }
        if (incoming == '\n') {
            if (commandOverflow) {
                Serial.println("error,command_too_long");
            } else if (commandLength > 0) {
                commandBuffer[commandLength] = '\0';
                processCommand(commandBuffer);
            }
            commandLength = 0;
            commandOverflow = false;
            continue;
        }

        if (commandOverflow) {
            continue;
        }
        if (commandLength + 1 >= COMMAND_BUFFER_SIZE) {
            commandOverflow = true;
            continue;
        }
        commandBuffer[commandLength++] = incoming;
    }
}

void serviceAlarm()
{
    noInterrupts();
    const uint32_t alarmCount = alarmEventsPending;
    alarmEventsPending = 0;
    interrupts();

    for (uint32_t i = 0; i < alarmCount; ++i) {
        Serial.println("alert");
        if (isMotionActive()) {
            stopMotion(true);
        }
    }
}

} // namespace

void setup()
{
    Serial.begin(115200);

    pinMode(DIRECTION_PIN, OUTPUT);
    pinMode(ENABLE_PIN, OUTPUT);
    pinMode(ALARM_PIN, INPUT_PULLDOWN);
    pinMode(PULSE_PIN, OUTPUT);

    digitalWrite(PULSE_PIN, LOW);
    digitalWrite(DIRECTION_PIN, HIGH);
    digitalWrite(ENABLE_PIN, LOW); // HBT4248C enabled with the existing wiring.

    attachInterrupt(digitalPinToInterrupt(ALARM_PIN), onAlarmRisingEdge, RISING);
    if (digitalRead(ALARM_PIN) == ALARM_ACTIVE_LEVEL) {
        alarmEventsPending = 1;
    }

    Serial.println("ready");
}

void loop()
{
    serviceSerial();
    serviceAlarm();
    serviceMotion();
}
