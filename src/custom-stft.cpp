#include <bit>
#include <cmath>
#include "esap/stft.hpp"

namespace esap {

/** @brief The number of bits required to represent half of the window size. */
constexpr usize HALF_WINDOW_SIZE_BITS = static_cast<usize>(
    std::countr_zero(Stft::HALF_WINDOW_SIZE)
);

/**
 * @brief Creates a buffer containing the bit-reversal permutation indices
 * for an FFT of size `Stft::HALF_WINDOW_SIZE`.
 *
 * @return A buffer containing the bit-reversal permutation indices.
 */
static constexpr std::array<usize, Stft::HALF_WINDOW_SIZE> make_bit_reversal() {
    std::array<usize, Stft::HALF_WINDOW_SIZE> bitReversal;
    for (usize i = 0; i < Stft::HALF_WINDOW_SIZE; i++) {
        usize j = 0;
        for (usize k = 0; k < HALF_WINDOW_SIZE_BITS; k++) {
            j = (j << 1) | ((i >> k) & 1);
        }
        bitReversal[i] = j;
    }
    return bitReversal;
}

/**
 * @brief Creates a buffer containing the twiddle factors for an FFT of size
 * `Stft::HALF_WINDOW_SIZE`.
 *
 * @param[in] inverse Whether to compute the twiddle factors for the forward
 *                    or inverse FFT.
 * @return A buffer containing the twiddle factors for the FFT.
 */
static constexpr std::array<std::complex<f32>, Stft::HALF_WINDOW_SIZE> make_twiddles(
    bool inverse = false
) {
    constexpr f32 PI = std::numbers::pi_v<f32>;
    std::array<std::complex<f32>, Stft::HALF_WINDOW_SIZE> twiddles;
    for (usize i = 0; i < Stft::HALF_WINDOW_SIZE; i++) {
        f32 angle = (inverse ? 2.0F : -2.0F) * PI * static_cast<f32>(i)
            / static_cast<f32>(Stft::WINDOW_SIZE);
        twiddles[i] = std::complex<f32>(std::cos(angle), std::sin(angle));
    }
    return twiddles;
}

struct Stft::Impl {

    /** @brief The imaginary unit as a complex number. */
    static constexpr std::complex<f32> I = { 0.0F, 1.0F };

    /** @brief The precomputed coefficients of a Hann window. */
    static constexpr std::array<f32, Stft::WINDOW_SIZE> WINDOW = make_window();

    /** @brief The precomputed bit-reversal permutation indices. */
    static constexpr std::array<usize, Stft::HALF_WINDOW_SIZE>
        BIT_REVERSAL = make_bit_reversal();

    /** @brief The precomputed twiddle factors for the forward FFT. */
    static constexpr std::array<std::complex<f32>, Stft::HALF_WINDOW_SIZE>
        FWD_TWIDDLES = make_twiddles(false);

    /** @brief The precomputed twiddle factors for the inverse FFT. */
    static constexpr std::array<std::complex<f32>, Stft::HALF_WINDOW_SIZE>
        INV_TWIDDLES = make_twiddles(true);

    /** @brief The format of the original audio signal. */
    AudioFormat _format;

    /** @brief The number of windowed FFTs computed per audio channel. */
    u32 _numFrames;

    /**
     * @brief The complex frequency bins.
     *
     * @note The bins are stored as three-dimensional array, where the first
     * dimension corresponds to the audio channel, the second dimension
     * corresponds to the frame, and the third dimension corresponds to the bin
     * (between `0` and `Stft::HALF_WINDOW_SIZE`, inclusively).
     */
    std::unique_ptr<std::complex<f32>[]> _bins;

    /**
     * @brief Computes the in-place complex-to-complex (C2C) Fast Fourier
     * Transform (FFT) of a buffer of complex samples.
     *
     * @param[in, out] buffer   The buffer of complex samples to transform.
     * @param[in]      twiddles Precomputed twiddle factors for the FFT.
     * @param[in]      inverse  Whether to compute the forward or inverse FFT.
     */
    static constexpr void fft_c2c(
        std::span<std::complex<f32>, Stft::HALF_WINDOW_SIZE> buffer,
        std::span<const std::complex<f32>, Stft::HALF_WINDOW_SIZE> twiddles,
        bool inverse = false
    ) noexcept {
        for (usize i = 0; i < Stft::HALF_WINDOW_SIZE; i++) {
            usize j = BIT_REVERSAL[i];
            if (i < j) {
                std::swap(buffer[i], buffer[j]);
            }
        }
        for (usize s = 1; s <= HALF_WINDOW_SIZE_BITS; s++) {
            usize m = static_cast<usize>(1) << s;
            usize stride = Stft::HALF_WINDOW_SIZE >> (s - 1);
            for (usize i = 0; i < Stft::HALF_WINDOW_SIZE; i += m) {
                for (usize j = 0; j < m / 2; j++) {
                    std::complex<f32> w = twiddles[j * stride];
                    std::complex<f32> t = w * buffer[i + j + m / 2];
                    std::complex<f32> u = buffer[i + j];
                    buffer[i + j] = u + t;
                    buffer[i + j + m / 2] = u - t;
                }
            }
        }
        if (inverse) {
            for (std::complex<f32>& x : buffer) {
                x /= static_cast<f32>(Stft::HALF_WINDOW_SIZE);
            }
        }
    }

    static std::unique_ptr<Impl> forward(
        const AudioFormat& format,
        std::span<const f32> samples
    ) {
        usize numFrames = (format.numFrames + Stft::HOP_SIZE - 1)
            / Stft::HOP_SIZE;
        auto bins = std::make_unique_for_overwrite<std::complex<f32>[]>(
            format.numChannels * numFrames * Stft::NUM_BINS
        );
        auto buffer = std::make_unique_for_overwrite<std::complex<f32>[]>(
            Stft::HALF_WINDOW_SIZE
        );
        for (usize c = 0; c < format.numChannels; c++) {
            for (usize f = 0; f < numFrames; f++) {
                for (usize i = 0; i < Stft::HALF_WINDOW_SIZE; i++) {
                    usize idx0 = f * Stft::HOP_SIZE + 2 * i;
                    usize idx1 = idx0 + 1;
                    f32 x0 = (idx0 < format.numFrames)
                        ? samples[c * format.numFrames + idx0]
                            * WINDOW[2 * i]
                        : 0.0F;
                    f32 x1 = (idx1 < format.numFrames)
                        ? samples[c * format.numFrames + idx1]
                            * WINDOW[2 * i + 1]
                        : 0.0F;
                    buffer[i] = std::complex<f32>(x0, x1);
                }
                fft_c2c(
                    std::span<std::complex<f32>, Stft::HALF_WINDOW_SIZE>(
                        buffer.get(),
                        Stft::HALF_WINDOW_SIZE
                    ),
                    FWD_TWIDDLES
                );
                usize idx = (c * numFrames + f) * Stft::NUM_BINS;
                bins[idx] = buffer[0].real() + buffer[0].imag();
                for (usize i = 0; i < Stft::HALF_WINDOW_SIZE; i++) {
                    std::complex<f32> zi = buffer[i];
                    std::complex<f32> zic = std::conj(
                        buffer[Stft::HALF_WINDOW_SIZE - i]
                    );
                    bins[idx + i] = 0.5F * (zi + zic)
                        - 0.5F * I * FWD_TWIDDLES[i] * (zi - zic);
                }
                bins[idx + Stft::HALF_WINDOW_SIZE] = buffer[0].real()
                    - buffer[0].imag();
            }
        }
        return std::make_unique<Impl>(
            format,
            static_cast<u32>(numFrames),
            std::move(bins)
        );
    }

    const AudioFormat& format() const noexcept {
        return _format;
    }

    u32 num_frames() const noexcept {
        return _numFrames;
    }

    void apply(std::span<const Filter> filters) {
        std::array<f32, Stft::NUM_BINS> gains = make_gains(_format, filters);
        for (usize c = 0; c < _format.numChannels; c++) {
            for (usize f = 0; f < _numFrames; f++) {
                for (usize i = 0; i < Stft::NUM_BINS; i++) {
                    usize idx = (c * _numFrames + f) * Stft::NUM_BINS + i;
                    _bins[idx] *= gains[i];
                }
            }
        }
    }

    std::span<const std::complex<f32>> bins() {
        return std::span<const std::complex<f32>>(
            _bins.get(),
            _format.numChannels * _numFrames * Stft::NUM_BINS
        );
    }

    std::unique_ptr<f32[]> inverse() const {
        auto frames = std::make_unique_for_overwrite<f32[]>(
            static_cast<usize>(_format.numChannels) * _numFrames
                * Stft::WINDOW_SIZE
        );
        auto buffer = std::make_unique_for_overwrite<std::complex<f32>[]>(
            Stft::HALF_WINDOW_SIZE
        );
        for (usize c = 0; c < _format.numChannels; c++) {
            for (usize f = 0; f < _numFrames; f++) {
                usize idx = (c * _numFrames + f) * Stft::NUM_BINS;
                for (usize i = 0; i < Stft::HALF_WINDOW_SIZE; i++) {
                    std::complex<f32> xi = _bins[idx + i];
                    std::complex<f32> xic = std::conj(
                        _bins[idx + Stft::HALF_WINDOW_SIZE - i]
                    );
                    buffer[i] = 0.5F * (xi + xic)
                        + 0.5F * I * INV_TWIDDLES[i] * (xi - xic);
                }
                fft_c2c(
                    std::span<std::complex<f32>, Stft::HALF_WINDOW_SIZE>(
                        buffer.get(),
                        Stft::HALF_WINDOW_SIZE
                    ),
                    INV_TWIDDLES,
                    true
                );
                for (usize i = 0; i < Stft::HALF_WINDOW_SIZE; i++) {
                    usize idx0 = (c * _numFrames + f) * Stft::WINDOW_SIZE
                        + 2 * i;
                    usize idx1 = idx0 + 1;
                    frames[idx0] = buffer[i].real() * WINDOW[2 * i];
                    frames[idx1] = buffer[i].imag() * WINDOW[2 * i + 1];
                }
            }
        }
        auto samples = std::make_unique<f32[]>(
            static_cast<usize>(_format.numChannels) * _format.numFrames
        );
        for (usize c = 0; c < _format.numChannels; c++) {
            for (usize f = 0; f < _format.numFrames; f++) {
                f32 sum = 0.0F;
                f32 norm = 0.0F;
                usize start = (f >= Stft::WINDOW_SIZE)
                    ? (f - Stft::WINDOW_SIZE) / Stft::HOP_SIZE + 1
                    : 0;
                for (usize i = start; i < _numFrames; i++) {
                    usize j = f - i * Stft::HOP_SIZE;
                    if (j >= Stft::WINDOW_SIZE) {
                        break;
                    }
                    sum += frames[(c * _numFrames + i) * Stft::WINDOW_SIZE + j];
                    norm += WINDOW[j] * WINDOW[j];
                }
                samples[c * _format.numFrames + f] = (norm > 1.0E-8F)
                    ? sum / norm
                    : 0.0F;
            }
        }
        return samples;
    }

};

Stft::Stft(std::unique_ptr<Stft::Impl> impl) noexcept
    : _impl(std::move(impl)) { }

Stft Stft::forward(const AudioFormat& format, std::span<const f32> samples) {
    return Stft(Stft::Impl::forward(format, samples));
}

Stft::Stft(Stft&& other) noexcept = default;

Stft& Stft::operator=(Stft&& other) noexcept = default;

const AudioFormat& Stft::format() const noexcept {
    return _impl->format();
}

u32 Stft::num_frames() const noexcept {
    return _impl->num_frames();
}

void Stft::apply(std::span<const Filter> filters) {
    _impl->apply(filters);
}

std::span<const std::complex<f32>> Stft::bins() {
    return _impl->bins();
}

std::unique_ptr<f32[]> Stft::inverse() const {
    return _impl->inverse();
}

Stft::~Stft() noexcept = default;

}