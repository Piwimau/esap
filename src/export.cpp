#include <cmath>
#include <cstdio>
#include <format>
#include <memory>
#include <print>
#include "esap/export.hpp"
#include "esap/func-deleter.hpp"

namespace esap {

/** @brief Represents a custom deleter for `std::FILE` handles. */
using FileDeleter = FuncDeleter<&std::fclose>;

void export_spectrum(
    usize numChannels,
    usize sampleRate,
    usize numFrames,
    usize windowSize,
    std::span<const std::complex<f32>> bins,
    const std::string& path
) {
    std::unique_ptr<std::FILE, FileDeleter> file(std::fopen(path.c_str(), "w"));
    if (file == nullptr) {
        throw std::runtime_error(std::format("Failed to open '{}'.", path));
    }
    std::println(file.get(), "channel,frequency_hz,magnitude_db");
    usize numBins = windowSize / 2 + 1;
    for (usize c = 0; c < numChannels; c++) {
        for (usize k = 0; k < numBins; k++) {
            f32 freq = static_cast<f32>(k) * static_cast<f32>(sampleRate)
                / static_cast<f32>(windowSize);
            f32 mag = 0.0F;
            for (usize f = 0; f < numFrames; f++) {
                usize idx = (c * numFrames + f) * numBins + k;
                mag += std::abs(bins[idx]);
            }
            mag /= static_cast<f32>(numFrames);
            f32 db = 20.0F * std::log10(
                mag / (static_cast<f32>(windowSize) / 4.0F) + 1.0E-10F
            );
            std::println(file.get(), "{},{},{}", c, freq, db);
        }
    }
}

void export_spectrogram(
    usize numChannels,
    usize sampleRate,
    usize numFrames,
    usize windowSize,
    usize hopSize,
    std::span<const std::complex<f32>> bins,
    const std::string& path
) {
    std::unique_ptr<std::FILE, FileDeleter> file(std::fopen(path.c_str(), "w"));
    if (file == nullptr) {
        throw std::runtime_error(std::format("Failed to open '{}'.", path));
    }
    std::print(file.get(), "channel,time_s");
    usize numBins = windowSize / 2 + 1;
    for (usize k = 0; k < numBins; k++) {
        f32 freq = static_cast<f32>(k) * static_cast<f32>(sampleRate)
            / static_cast<f32>(windowSize);
        std::print(file.get(), ",{}hz", freq);
    }
    std::println(file.get());
    for (usize c = 0; c < numChannels; c++) {
        for (usize f = 0; f < numFrames; f++) {
            f32 time = static_cast<f32>(f) * static_cast<f32>(hopSize)
                / static_cast<f32>(sampleRate);
            std::print(file.get(), "{},{}", c, time);
            for (usize k = 0; k < numBins; k++) {
                usize idx = (c * numFrames + f) * numBins + k;
                f32 mag = std::abs(bins[idx]);
                f32 db = 20.0F * std::log10(
                    mag / (static_cast<f32>(windowSize) / 4.0F) + 1.0E-10F
                );
                std::print(file.get(), ",{}", db);
            }
            std::println(file.get());
        }
    }
}

}