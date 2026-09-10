#ifndef ESAP_STFT_HPP
#define ESAP_STFT_HPP

#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <numbers>
#include <span>
#include "esap/audio-format.hpp"
#include "esap/filter.hpp"
#ifdef ESAP_USE_OPENCL
    #include "esap/gpu-context.hpp"
#endif
#include "esap/types.hpp"

namespace esap {

/**
 * @brief Represents a Short-Time Fourier Transform (STFT) of an audio signal.
 */
class Stft final {
private:

    /** @brief Represents a concrete implementation of the STFT. */
    struct Impl;

    /** @brief The concrete implementation of the STFT. */
    std::unique_ptr<Impl> _impl;

    /**
     * @brief Initializes a new Short-Time Fourier Transform (STFT) with the
     * specified concrete implementation.
     *
     * @param[in] impl The concrete implementation of the STFT.
     */
    explicit Stft(std::unique_ptr<Impl> impl) noexcept;

public:

    /** @brief The number of audio samples per window. */
    static constexpr usize WINDOW_SIZE = 2048;

    /** @brief Half of the window size. */
    static constexpr usize HALF_WINDOW_SIZE = WINDOW_SIZE / 2;

    /** @brief The number of audio samples to advance per window. */
    static constexpr usize HOP_SIZE = HALF_WINDOW_SIZE;

    /** @brief The number of frequency bins per window. */
    static constexpr usize NUM_BINS = HALF_WINDOW_SIZE + 1;

    /**
     * @brief Computes the Short-Time Fourier Transform (STFT) of an audio
     * signal.
     *
     * @note Some implementations may leverage GPU acceleration if a valid
     * GPU context is provided as an argument, which must remain valid for the
     * lifetime of the `Stft` object returned by this function. Note that the
     * GPU context may be modified during a call to a method of this class.
     *
     * @param[in]      format     The format of the original audio signal.
     * @param[in]      samples    The time-domain audio samples to transform.
     * @param[in, out] gpuContext A GPU context that may be used to accelerate
     *                            the computation.
     * @return The Short-Time Fourier Transform (STFT) of the audio signal.
     */
    static Stft forward(
        const AudioFormat& format,
        std::span<const f32> samples
    #ifdef ESAP_USE_OPENCL
        , GpuContext* gpuContext
    #endif
    );

    Stft(const Stft&) = delete;

    Stft& operator=(const Stft&) = delete;

    Stft(Stft&&) noexcept;

    Stft& operator=(Stft&&) noexcept;

    /**
     * @brief Returns the format of the original audio signal.
     *
     * @return The format of the original audio signal.
     */
    const AudioFormat& format() const noexcept;

    /**
     * @brief Returns the number of windowed FFTs computed per audio channel.
     *
     * @return The number of windowed FFTs computed per audio channel.
     */
    u32 num_frames() const noexcept;

    /**
     * @brief Applies a sequence of filters.
     *
     * @param[in] filters The filters to apply.
     */
    void apply(std::span<const Filter> filters);

    /**
     * @brief Returns a read-only view of the complex frequency bins.
     *
     * @return A read-only view of the complex frequency bins.
     */
    std::span<const std::complex<f32>> bins();

    /**
     * @brief Computes the inverse Short-Time Fourier Transform (ISTFT) to
     * reconstruct the time-domain audio signal.
     *
     * @return The reconstructed time-domain audio samples.
     */
    std::unique_ptr<f32[]> inverse() const;

    ~Stft() noexcept;

};

/**
 * @brief Creates a buffer containing the coefficients of a Hann window.
 *
 * @return A buffer containing the coefficients of a Hann window.
 */
constexpr std::array<f32, Stft::WINDOW_SIZE> make_window() {
    constexpr f32 PI = std::numbers::pi_v<f32>;
    std::array<f32, Stft::WINDOW_SIZE> window;
    for (usize i = 0; i < Stft::WINDOW_SIZE; i++) {
        window[i] = 0.5F
            * (1.0F
               - std::cos(2.0F * PI * static_cast<f32>(i) / Stft::WINDOW_SIZE));
    }
    return window;
}

/**
 * @brief Creates a buffer containing the frequency-domain gains for the
 * specified audio format and sequence of filters.
 *
 * @param[in] format  The format of the original audio signal.
 * @param[in] filters The sequence of filters to apply.
 * @return A buffer containing the frequency-domain gains for the specified
 * audio format and sequence of filters.
 */
constexpr std::array<f32, Stft::NUM_BINS> make_gains(
    const AudioFormat& format,
    std::span<const Filter> filters
) {
    std::array<f32, Stft::NUM_BINS> gains;
    std::ranges::fill_n(gains.begin(), Stft::NUM_BINS, 1.0F);
    for (const Filter& filter : filters) {
        std::visit(
            [&](const auto& f) {
                using T = std::decay_t<decltype(f)>;
                for (usize i = 0; i < Stft::NUM_BINS; i++) {
                    f32 freq = static_cast<f32>(i)
                        * static_cast<f32>(format.sampleRate)
                        / static_cast<f32>(Stft::WINDOW_SIZE);
                    if constexpr (std::is_same_v<T, LowpassFilter>) {
                        gains[i] *= (freq < f.cutoff) ? 1.0F : 0.0F;
                    }
                    else if constexpr (std::is_same_v<T, HighpassFilter>) {
                        gains[i] *= (freq > f.cutoff) ? 1.0F : 0.0F;
                    }
                    else if constexpr (std::is_same_v<T, BandpassFilter>) {
                        gains[i] *= ((freq > f.lowCutoff) && (freq < f.highCutoff))
                            ? 1.0F
                            : 0.0F;
                    }
                    else {
                        gains[i] *= ((freq < f.lowCutoff) || (freq > f.highCutoff))
                            ? 1.0F
                            : 0.0F;
                    }
                }
            },
            filter
        );
    }
    return gains;
}

}

#endif