#include <cassert>
#include <cmath>
#include <cstdint>

#include "../src/motion_math.h"

int main()
{
    constexpr uint32_t pulsesPerRevolution = 52100;

    assert(!motion::isValidIncrement(0.006, pulsesPerRevolution));
    assert(motion::isValidIncrement(0.007, pulsesPerRevolution));
    assert(motion::isValidIncrement(5.0, pulsesPerRevolution));
    assert(motion::isValidIncrement(360.0, pulsesPerRevolution));
    assert(!motion::isValidIncrement(360.01, pulsesPerRevolution));

    assert(motion::cumulativePulseTarget(5.0, pulsesPerRevolution) == 724);
    assert(motion::cumulativePulseTarget(10.0, pulsesPerRevolution) == 1447);
    assert(motion::cumulativePulseTarget(360.0, pulsesPerRevolution) == 52100);
    assert(motion::additionalPulsesToTarget(52100, 52100) == 0);
    assert(motion::additionalPulsesToTarget(51376, 52100) == 724);

    const double startRateHz = 0.5 * pulsesPerRevolution / 60.0;
    const double maximumRateHz = 2.0 * pulsesPerRevolution / 60.0;
    const double accelerationHzPerSecond =
        8.0 * pulsesPerRevolution / 60.0;

    const double firstPulseRate = motion::sCurvePulseRateHz(
        0, 724, startRateHz, maximumRateHz, accelerationHzPerSecond);
    const double earlyPulseRate = motion::sCurvePulseRateHz(
        20, 724, startRateHz, maximumRateHz, accelerationHzPerSecond);
    const double latePulseRate = motion::sCurvePulseRateHz(
        703, 724, startRateHz, maximumRateHz, accelerationHzPerSecond);
    const double middlePulseRate = motion::sCurvePulseRateHz(
        362, 724, startRateHz, maximumRateHz, accelerationHzPerSecond);
    const double lastPulseRate = motion::sCurvePulseRateHz(
        723, 724, startRateHz, maximumRateHz, accelerationHzPerSecond);

    assert(std::fabs(firstPulseRate - startRateHz) < 1e-9);
    assert(earlyPulseRate > startRateHz);
    assert(earlyPulseRate < middlePulseRate);
    assert(std::fabs(earlyPulseRate - latePulseRate) < 1e-9);
    assert(std::fabs(middlePulseRate - maximumRateHz) < 1e-9);
    assert(std::fabs(lastPulseRate - startRateHz) < 1e-9);

    const uint32_t brakingPulses = motion::brakingPulseCount(
        maximumRateHz, startRateHz, accelerationHzPerSecond);
    assert(brakingPulses == 204);
    const double firstBrakingRate = motion::sCurveStopRateHz(
        0, brakingPulses, maximumRateHz, startRateHz);
    const double finalBrakingRate = motion::sCurveStopRateHz(
        brakingPulses - 1, brakingPulses, maximumRateHz, startRateHz);
    assert(firstBrakingRate < maximumRateHz);
    assert(firstBrakingRate > startRateHz);
    assert(std::fabs(finalBrakingRate - startRateHz) < 1e-9);
    assert(motion::pulseHalfPeriodUs(maximumRateHz) == 288);

    assert(std::fabs(motion::nextSegmentDegrees(0.0, 5.0) - 5.0) < 1e-9);
    assert(std::fabs(motion::nextSegmentDegrees(357.0, 7.0) - 3.0) < 1e-9);

    double completedDegrees = 0.0;
    uint32_t emittedPulses = 0;
    while (completedDegrees < 360.0) {
        const double segment = motion::nextSegmentDegrees(completedDegrees, 5.0);
        const double targetDegrees = completedDegrees + segment;
        const uint32_t targetPulses =
            motion::cumulativePulseTarget(targetDegrees, pulsesPerRevolution);
        emittedPulses += targetPulses - emittedPulses;
        completedDegrees = targetDegrees;
    }

    assert(emittedPulses == pulsesPerRevolution);
    return 0;
}
