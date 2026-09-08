#ifndef ESAP_BENCH_HPP
#define ESAP_BENCH_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>
#include "esap/types.hpp"

namespace esap {

/** @brief Represents a benchmark statistic. */
struct BenchStats {

    /** @brief The minimum observed time across all iterations. */
    std::chrono::nanoseconds min;

    /** @brief The maximum observed time across all iterations. */
    std::chrono::nanoseconds max;

    /** @brief The mean observed time across all iterations. */
    std::chrono::nanoseconds mean;

    /** @brief The median observed time across all iterations. */
    std::chrono::nanoseconds median;

    /**
     * @brief The standard deviation of observed times across all iterations.
     */
    std::chrono::nanoseconds stdDev;

};

/** @brief Represents the result of a benchmark. */
struct BenchResult {

    /** @brief The number of warmup iterations performed. */
    usize warmups;

    /** @brief The number of benchmark iterations performed. */
    usize iterations;

    /** @brief The benchmark statistics for the wall time. */
    BenchStats wall;

    /** @brief The benchmark statistics for the CPU time. */
    BenchStats cpu;

};

/**
 * @brief Returns the current CPU time of the calling thread in nanoseconds.
 *
 * @note This function is an implementation detail and is not intended to be
 * called directly.
 *
 * @return The current CPU time of the calling thread in nanoseconds on success,
 * or `std::nullopt` on failure.
 */
std::optional<std::chrono::nanoseconds> cpu_time();

/**
 * @brief Benchmarks a specified function by performing a specified number of
 * warmup and benchmark iterations and collecting timing statistics.
 *
 * @param[in] warmups    The number of warmup iterations to perform before
 *                       benchmarking.
 * @param[in] iterations The number of benchmark iterations to perform.
 * @param[in] func       The function to benchmark.
 * @param[in] args       The arguments to pass to the function being benchmarked
 *                       (if any).
 * @return A benchmark result containing the collected timing statistics for the
 * wall and CPU times.
 * @throws `std::runtime_error` If an error occurs while retrieving the CPU time
 *                              during benchmarking.
 */
template<typename Func, typename... Args>
requires std::invocable<Func, Args...>
BenchResult benchmark(
    usize warmups,
    usize iterations,
    Func&& func,
    Args&&... args
) {
    auto call = [&]() { std::invoke(func, args...); };
    for (usize i = 0; i < warmups; i++) {
        call();
    }
    std::vector<std::chrono::nanoseconds> wallSamples(iterations);
    std::vector<std::chrono::nanoseconds> cpuSamples(iterations);
    for (usize i = 0; i < iterations; i++) {
        auto cpuStart = cpu_time();
        auto wallStart = std::chrono::high_resolution_clock::now();
        if (!cpuStart) {
            throw std::runtime_error("Failed to get the CPU time.");
        }
        call();
        auto wallEnd = std::chrono::high_resolution_clock::now();
        auto cpuEnd = cpu_time();
        if (!cpuEnd) {
            throw std::runtime_error("Failed to get the CPU time.");
        }
        wallSamples[i] = wallEnd - wallStart;
        cpuSamples[i] = *cpuEnd - *cpuStart;
    }
    auto compute_stats = [](
        std::span<std::chrono::nanoseconds> samples
    ) -> BenchStats {
        std::ranges::sort(samples);
        std::chrono::nanoseconds min = samples.front();
        std::chrono::nanoseconds max = samples.back();
        usize n = samples.size();
        std::chrono::nanoseconds median = ((n % 2) == 0)
            ? (samples[n / 2 - 1] + samples[n / 2]) / 2
            : samples[n / 2];
        i64 sum = 0;
        for (const std::chrono::nanoseconds& sample : samples) {
            sum += sample.count();
        }
        std::chrono::nanoseconds mean(sum / n);
        f64 variance = 0.0;
        for (const std::chrono::nanoseconds& sample : samples) {
            f64 diff = static_cast<f64>(sample.count() - mean.count());
            variance += diff * diff;
        }
        variance /= static_cast<f64>(n);
        std::chrono::nanoseconds stdDev(static_cast<i64>(std::sqrt(variance)));
        return {
            .min = min,
            .max = max,
            .mean = mean,
            .median = median,
            .stdDev = stdDev
        };
    };
    return {
        .warmups = warmups,
        .iterations = iterations,
        .wall = compute_stats(wallSamples),
        .cpu = compute_stats(cpuSamples)
    };
}

}

#endif