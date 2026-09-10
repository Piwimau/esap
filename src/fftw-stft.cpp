#include <algorithm>
#include <fftw3.h>
#include <type_traits>
#include "esap/func-deleter.hpp"
#include "esap/stft.hpp"

namespace esap {

struct Stft::Impl {

    /** @brief Represents a custom deleter for memory allocated by FFTW. */
    using FftwFreeDeleter = FuncDeleter<&fftwf_free>;

    /** @brief Represents a custom deleter for FFTW plans. */
    using FftwPlanDeleter = FuncDeleter<&fftwf_destroy_plan>;

    /** @brief The precomputed coefficients of a Hann window. */
    static constexpr std::array<f32, Stft::WINDOW_SIZE> WINDOW = make_window();

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

    static std::unique_ptr<Impl> forward(
        const AudioFormat& format,
        std::span<const f32> samples
    ) {
        usize numFrames = (format.numFrames + Stft::HOP_SIZE - 1)
            / Stft::HOP_SIZE;
        auto bins = std::make_unique_for_overwrite<std::complex<f32>[]>(
            format.numChannels * numFrames * Stft::NUM_BINS
        );
        std::unique_ptr<f32[], FftwFreeDeleter> input(
            fftwf_alloc_real(Stft::WINDOW_SIZE)
        );
        std::unique_ptr<std::complex<f32>[], FftwFreeDeleter> output(
            reinterpret_cast<std::complex<f32>*>(
                fftwf_alloc_complex(Stft::NUM_BINS)
            )
        );
        std::unique_ptr<std::remove_pointer_t<fftwf_plan>, FftwPlanDeleter> plan(
            fftwf_plan_dft_r2c_1d(
                static_cast<int>(Stft::WINDOW_SIZE),
                input.get(),
                reinterpret_cast<fftwf_complex*>(output.get()),
                FFTW_ESTIMATE
            )
        );
        for (usize c = 0; c < format.numChannels; c++) {
            for (usize f = 0; f < numFrames; f++) {
                for (usize i = 0; i < Stft::WINDOW_SIZE; i++) {
                    usize idx = f * Stft::HOP_SIZE + i;
                    input[i] = (idx < format.numFrames)
                        ? samples[c * format.numFrames + idx] * WINDOW[i]
                        : 0.0F;
                }
                fftwf_execute(plan.get());
                std::ranges::copy_n(
                    output.get(),
                    Stft::NUM_BINS,
                    bins.get() + (c * numFrames + f) * Stft::NUM_BINS
                );
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
        std::unique_ptr<std::complex<f32>[], FftwFreeDeleter> input(
            reinterpret_cast<std::complex<f32>*>(
                fftwf_alloc_complex(Stft::NUM_BINS)
            )
        );
        std::unique_ptr<f32[], FftwFreeDeleter> output(
            fftwf_alloc_real(Stft::WINDOW_SIZE)
        );
        std::unique_ptr<std::remove_pointer_t<fftwf_plan>, FftwPlanDeleter> plan(
            fftwf_plan_dft_c2r_1d(
                static_cast<int>(Stft::WINDOW_SIZE),
                reinterpret_cast<fftwf_complex*>(input.get()),
                output.get(),
                FFTW_MEASURE
            )
        );
        for (usize c = 0; c < _format.numChannels; c++) {
            for (usize f = 0; f < _numFrames; f++) {
                std::ranges::copy_n(
                    _bins.get() + (c * _numFrames + f) * Stft::NUM_BINS,
                    Stft::NUM_BINS,
                    input.get()
                );
                fftwf_execute(plan.get());
                for (usize i = 0; i < Stft::WINDOW_SIZE; i++) {
                    usize idx = (c * _numFrames + f) * Stft::WINDOW_SIZE + i;
                    frames[idx] = output[i] / Stft::WINDOW_SIZE * WINDOW[i];
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