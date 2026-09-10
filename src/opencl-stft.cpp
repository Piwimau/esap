#include <cassert>
#include <concepts>
#include "esap/stft.hpp"

namespace esap {

struct Stft::Impl {

    /** @brief An appropriate size for local workgroups. */
    static constexpr usize NUM_WORKERS = 256;

    /**
     * @brief Rounds up a value to the nearest multiple of another value.
     *
     * @tparam T The type of the value to round up.
     * @tparam U The type of the multiple to round up to.
     * @param[in] value    The value to round up.
     * @param[in] multiple The multiple to round up to.
     * @return The result of rounding up `value` to the nearest multiple of
     * `multiple`.
     */
    template<std::unsigned_integral T, std::unsigned_integral U>
    static constexpr auto round_up(T value, U multiple) noexcept {
        return (value + multiple - 1) / multiple * multiple;
    }

    /** @brief The format of the original audio signal. */
    AudioFormat _format;

    /** @brief The number of windowed FFTs computed per audio channel. */
    u32 _numFrames;

    /** @brief A GPU context used to accelerate the computation. */
    GpuContext* _gpuContext;

    /**
     * @brief The complex frequency bins.
     *
     * @note The bins are stored as three-dimensional array, where the first
     * dimension corresponds to the audio channel, the second dimension
     * corresponds to the frame, and the third dimension corresponds to the bin
     * (between `0` and `Stft::HALF_WINDOW_SIZE`, inclusively).
     */
    std::unique_ptr<std::complex<f32>[]> _bins;

    /** @brief The GPU-side buffer containing the complex frequency bins. */
    cl::Buffer _gpuBins;

    static std::unique_ptr<Impl> forward(
        const AudioFormat& format,
        std::span<const f32> samples,
        GpuContext* gpuContext
    ) {
        assert(gpuContext != nullptr);
        std::unique_ptr<std::complex<f32>[]> bins;
        cl::Buffer gpuSamples(
            gpuContext->context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            format.numChannels * format.numFrames * sizeof(f32),
            const_cast<f32*>(samples.data())
        );
        usize numFrames = (format.numFrames + Stft::HOP_SIZE - 1)
            / Stft::HOP_SIZE;
        usize gpuInputBufferSize = format.numChannels * numFrames
            * Stft::WINDOW_SIZE * sizeof(f32);
        cl::Buffer gpuInputBuffer(
            gpuContext->context,
            CL_MEM_READ_WRITE,
            gpuInputBufferSize
        );
        cl_mem rawGpuInputBuffer = gpuInputBuffer.get();
        gpuContext->applyWindow.setArg(0, gpuInputBuffer);
        gpuContext->applyWindow.setArg(1, gpuSamples);
        gpuContext->applyWindow.setArg(2, gpuContext->windowBuffer);
        gpuContext->applyWindow.setArg(3, static_cast<cl_ulong>(numFrames));
        gpuContext->applyWindow.setArg(
            4,
            static_cast<cl_ulong>(format.numFrames)
        );
        gpuContext->queue.enqueueNDRangeKernel(
            gpuContext->applyWindow,
            cl::NullRange,
            cl::NDRange(
                format.numChannels * numFrames,
                round_up(Stft::WINDOW_SIZE, NUM_WORKERS)
            ),
            cl::NDRange(1, NUM_WORKERS)
        );
        usize gpuOutputBufferSize = format.numChannels * numFrames
            * Stft::NUM_BINS * sizeof(std::complex<f32>);
        cl::Buffer gpuOutputBuffer(
            gpuContext->context,
            CL_MEM_READ_WRITE,
            gpuOutputBufferSize
        );
        cl_mem rawGpuOutputBuffer = gpuOutputBuffer.get();
        if (format.numChannels * numFrames != gpuContext->numFwdBatches) {
            VkFFTConfiguration config = { };
            config.makeForwardPlanOnly = 1;
            config.FFTdim = 2;
            config.size[0] = Stft::WINDOW_SIZE;
            config.size[1] = format.numChannels * numFrames;
            config.omitDimension[1] = 1;
            config.performR2C = 1;
            config.device = &gpuContext->rawDevice;
            config.context = &gpuContext->rawContext;
            config.isInputFormatted = 1;
            config.inputBuffer = &rawGpuInputBuffer;
            config.inputBufferSize = &gpuInputBufferSize;
            config.buffer = &rawGpuOutputBuffer;
            config.bufferSize = &gpuOutputBufferSize;
            gpuContext->numFwdBatches = format.numChannels * numFrames;
            gpuContext->fwdApp.reset(new VkFFTApplication());
            VkFFTResult result = initializeVkFFT(
                gpuContext->fwdApp.get(),
                config
            );
            if (result != VKFFT_SUCCESS) {
                throw std::runtime_error(
                    std::format(
                        "Failed to initialize VkFFT: {}.",
                        getVkFFTErrorString(result)
                    )
                );
            }
        }
        VkFFTLaunchParams params = { };
        params.commandQueue = &gpuContext->rawQueue;
        params.inputBuffer = &rawGpuInputBuffer;
        params.buffer = &rawGpuOutputBuffer;
        VkFFTResult result = VkFFTAppend(
            gpuContext->fwdApp.get(),
            -1,
            &params
        );
        if (result != VKFFT_SUCCESS) {
            throw std::runtime_error(
                std::format(
                    "Failed to execute VkFFT: {}.",
                    getVkFFTErrorString(result)
                )
            );
        }
        gpuContext->queue.finish();
        return std::make_unique<Impl>(
            format,
            static_cast<u32>(numFrames),
            gpuContext,
            std::move(bins),
            gpuOutputBuffer
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
        cl::Buffer gpuGains(
            _gpuContext->context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            Stft::NUM_BINS * sizeof(f32),
            gains.data()
        );
        _gpuContext->applyGains.setArg(0, _gpuBins);
        _gpuContext->applyGains.setArg(1, gpuGains);
        _gpuContext->queue.enqueueNDRangeKernel(
            _gpuContext->applyGains,
            cl::NullRange,
            cl::NDRange(
                _format.numChannels * _numFrames,
                round_up(Stft::NUM_BINS, NUM_WORKERS)
            ),
            cl::NDRange(1, NUM_WORKERS)
        );
        _gpuContext->queue.finish();
        _bins = nullptr;
    }

    std::span<const std::complex<f32>> bins() {
        if (_bins == nullptr) {
            _bins = std::make_unique_for_overwrite<std::complex<f32>[]>(
                _format.numChannels * _numFrames * Stft::NUM_BINS
            );
            _gpuContext->queue.enqueueReadBuffer(
                _gpuBins,
                CL_TRUE,
                0,
                _format.numChannels * _numFrames * Stft::NUM_BINS
                    * sizeof(std::complex<f32>),
                _bins.get()
            );
        }
        return std::span<const std::complex<f32>>(
            _bins.get(),
            _format.numChannels * _numFrames * Stft::NUM_BINS
        );
    }

    std::unique_ptr<f32[]> inverse() const {
        usize numFrames = (_format.numFrames + Stft::HOP_SIZE - 1)
            / Stft::HOP_SIZE;
        usize gpuInputBufferSize = _format.numChannels * numFrames
            * Stft::NUM_BINS * sizeof(std::complex<f32>);
        cl_mem rawGpuInputBuffer = _gpuBins.get();
        usize gpuOutputBufferSize = _format.numChannels * numFrames
            * Stft::WINDOW_SIZE * sizeof(f32);
        cl::Buffer gpuOutputBuffer(
            _gpuContext->context,
            CL_MEM_READ_WRITE,
            gpuOutputBufferSize
        );
        cl_mem rawGpuOutputBuffer = gpuOutputBuffer.get();
        if (_format.numChannels * numFrames != _gpuContext->numInvBatches) {
            VkFFTConfiguration config = { };
            config.makeInversePlanOnly = 1;
            config.FFTdim = 2;
            config.size[0] = Stft::WINDOW_SIZE;
            config.size[1] = _format.numChannels * numFrames;
            config.omitDimension[1] = 1;
            config.performR2C = 1;
            config.device = &_gpuContext->rawDevice;
            config.context = &_gpuContext->rawContext;
            config.buffer = &rawGpuInputBuffer;
            config.bufferSize = &gpuInputBufferSize;
            config.isOutputFormatted = 1;
            config.outputBuffer = &rawGpuOutputBuffer;
            config.outputBufferSize = &gpuOutputBufferSize;
            _gpuContext->numInvBatches = _format.numChannels * numFrames;
            _gpuContext->invApp.reset(new VkFFTApplication());
            VkFFTResult result = initializeVkFFT(
                _gpuContext->invApp.get(),
                config
            );
            if (result != VKFFT_SUCCESS) {
                throw std::runtime_error(
                    std::format(
                        "Failed to initialize VkFFT: {}.",
                        getVkFFTErrorString(result)
                    )
                );
            }
        }
        VkFFTLaunchParams params = { };
        params.commandQueue = &_gpuContext->rawQueue;
        params.buffer = &rawGpuInputBuffer;
        params.outputBuffer = &rawGpuOutputBuffer;
        VkFFTResult result = VkFFTAppend(_gpuContext->invApp.get(), 1, &params);
        if (result != VKFFT_SUCCESS) {
            throw std::runtime_error(
                std::format(
                    "Failed to execute VkFFT: {}.",
                    getVkFFTErrorString(result)
                )
            );
        }
        auto samples = std::make_unique<f32[]>(
            static_cast<usize>(_format.numChannels) * _format.numFrames
        );
        cl::Buffer gpuSamples(
            _gpuContext->context,
            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
            _format.numChannels * _format.numFrames * sizeof(f32)
        );
        _gpuContext->overlapAdd.setArg(0, gpuSamples);
        _gpuContext->overlapAdd.setArg(1, gpuOutputBuffer);
        _gpuContext->overlapAdd.setArg(2, _gpuContext->windowBuffer);
        _gpuContext->overlapAdd.setArg(3, static_cast<cl_ulong>(_numFrames));
        _gpuContext->overlapAdd.setArg(
            4,
            static_cast<cl_ulong>(_format.numFrames)
        );
        _gpuContext->queue.enqueueNDRangeKernel(
            _gpuContext->overlapAdd,
            cl::NullRange,
            cl::NDRange(
                _format.numChannels,
                round_up(_format.numFrames, NUM_WORKERS)
            ),
            cl::NDRange(1, NUM_WORKERS)
        );
        _gpuContext->queue.enqueueReadBuffer(
            gpuSamples,
            CL_TRUE,
            0,
            _format.numChannels * _format.numFrames * sizeof(f32),
            samples.get()
        );
        return samples;
    }

};

Stft::Stft(std::unique_ptr<Stft::Impl> impl) noexcept
    : _impl(std::move(impl)) { }

Stft Stft::forward(
    const AudioFormat& format,
    std::span<const f32> samples,
    GpuContext* gpuContext
) {
    return Stft(Stft::Impl::forward(format, samples, gpuContext));
}

const AudioFormat& Stft::format() const noexcept {
    return _impl->format();
}

Stft::Stft(Stft&& other) noexcept = default;

Stft& Stft::operator=(Stft&& other) noexcept = default;

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