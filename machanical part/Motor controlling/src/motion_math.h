#pragma once

#include <cmath>
#include <cstdint>

namespace motion {

constexpr double FULL_REVOLUTION_DEGREES = 360.0;

inline bool isValidIncrement(double incrementDegrees, uint32_t pulsesPerRevolution)
{
    if (!std::isfinite(incrementDegrees) || pulsesPerRevolution == 0) {
        return false;
    }

    const double minimumIncrement = FULL_REVOLUTION_DEGREES / pulsesPerRevolution;
    return incrementDegrees + 1e-9 >= minimumIncrement
        && incrementDegrees <= FULL_REVOLUTION_DEGREES;
}

inline uint32_t cumulativePulseTarget(double completedDegrees,
                                      uint32_t pulsesPerRevolution)
{
    const double boundedDegrees =
        std::fmax(0.0, std::fmin(completedDegrees, FULL_REVOLUTION_DEGREES));
    return static_cast<uint32_t>(std::lround(
        boundedDegrees * pulsesPerRevolution / FULL_REVOLUTION_DEGREES));
}

inline uint32_t continuousPulseTarget(double degrees,
                                      uint32_t pulsesPerRevolution)
{
    if (!std::isfinite(degrees) || degrees <= 0.0
        || pulsesPerRevolution == 0) {
        return 0;
    }
    return static_cast<uint32_t>(std::lround(
        degrees * pulsesPerRevolution / FULL_REVOLUTION_DEGREES));
}

inline double continuousRunoutDegrees(double incrementDegrees)
{
    return FULL_REVOLUTION_DEGREES + incrementDegrees;
}

inline uint32_t additionalPulsesToTarget(uint32_t emittedPulses,
                                         uint32_t targetPulses)
{
    return targetPulses > emittedPulses ? targetPulses - emittedPulses : 0;
}

inline double smootherstep(double progress)
{
    const double bounded = std::fmax(0.0, std::fmin(progress, 1.0));
    return bounded * bounded * bounded
        * (bounded * (bounded * 6.0 - 15.0) + 10.0);
}

inline uint32_t brakingPulseCount(double currentRateHz,
                                  double stopRateHz,
                                  double accelerationHzPerSecond)
{
    if (currentRateHz <= stopRateHz || stopRateHz <= 0.0
        || accelerationHzPerSecond <= 0.0) {
        return 0;
    }
    return static_cast<uint32_t>(std::ceil(
        (currentRateHz * currentRateHz - stopRateHz * stopRateHz)
        / (2.0 * accelerationHzPerSecond)));
}

inline double sCurvePulseRateHz(uint32_t completedSegmentPulses,
                                uint32_t totalSegmentPulses,
                                double startRateHz,
                                double maximumRateHz,
                                double accelerationHzPerSecond)
{
    if (totalSegmentPulses == 0 || startRateHz <= 0.0
        || maximumRateHz < startRateHz || accelerationHzPerSecond <= 0.0) {
        return 0.0;
    }

    if (totalSegmentPulses == 1) {
        return startRateHz;
    }

    const double lastPulseIndex = totalSegmentPulses - 1.0;
    const double pulsesToMaximum =
        (maximumRateHz * maximumRateHz - startRateHz * startRateHz)
        / (2.0 * accelerationHzPerSecond);
    const double rampSpan = std::fmin(pulsesToMaximum, lastPulseIndex / 2.0);
    const double peakRateHz = std::fmin(
        maximumRateHz,
        std::sqrt(startRateHz * startRateHz
                  + 2.0 * accelerationHzPerSecond * rampSpan));

    const double pulseIndex = std::fmin(
        static_cast<double>(completedSegmentPulses), lastPulseIndex);
    double rampProgress = 1.0;
    if (rampSpan > 0.0 && pulseIndex < rampSpan) {
        rampProgress = pulseIndex / rampSpan;
    } else if (rampSpan > 0.0 && pulseIndex > lastPulseIndex - rampSpan) {
        rampProgress = (lastPulseIndex - pulseIndex) / rampSpan;
    }

    return startRateHz
        + (peakRateHz - startRateHz) * smootherstep(rampProgress);
}

inline double sCurveStopRateHz(uint32_t completedBrakingPulses,
                               uint32_t totalBrakingPulses,
                               double initialRateHz,
                               double stopRateHz)
{
    if (totalBrakingPulses == 0 || initialRateHz <= stopRateHz) {
        return stopRateHz;
    }
    const double progress = std::fmin(
        static_cast<double>(completedBrakingPulses + 1) / totalBrakingPulses,
        1.0);
    return initialRateHz
        + (stopRateHz - initialRateHz) * smootherstep(progress);
}

inline uint32_t pulseHalfPeriodUs(double pulseRateHz)
{
    if (!std::isfinite(pulseRateHz) || pulseRateHz <= 0.0) {
        return 0;
    }
    return static_cast<uint32_t>(std::lround(500000.0 / pulseRateHz));
}

inline double nextSegmentDegrees(double completedDegrees, double incrementDegrees)
{
    const double remainingDegrees =
        std::fmax(0.0, FULL_REVOLUTION_DEGREES - completedDegrees);
    return std::fmin(incrementDegrees, remainingDegrees);
}

} // namespace motion
