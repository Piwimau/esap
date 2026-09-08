#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <format>
#include <numbers>
#include <type_traits>
#include <vector>
#include "esap/stft.hpp"
#ifdef IMPL_FFTW
    #include <fftw3.h>
#endif
#include "esap/file-deleter.hpp"

namespace esap {

/** @brief The value of pi as a 32-bit floating-point number. */
static constexpr f32 PI = std::numbers::pi_v<f32>;

#ifdef IMPL_CUSTOM
    /** @brief The imaginary unit as a complex number. */
    static constexpr std::complex<f32> I(0.0F, 1.0F);
#endif

/** @brief The fraction of the bandwidth to use for the rolloff. */
static constexpr f32 ROLLOFF_FRAC = 0.01F;

/**
 * @brief Returns an array with the coefficients of a Hann window.
 *
 * @return An array with the coefficients of a Hann window.
 */
static constexpr inline std::array<f32, Stft::WINDOW_SIZE> make_window() {
    std::array<f32, Stft::WINDOW_SIZE> window;
    for (usize i = 0; i < Stft::WINDOW_SIZE; i++) {
        window[i] = 0.5F
            * (1.0F
                - std::cos(
                    2.0F * PI * static_cast<f32>(i)
                    / static_cast<f32>(Stft::WINDOW_SIZE)
                ));
    }
    return window;
}

/** @brief The precomputed coefficients of a Hann window. */
static constexpr std::array<f32, Stft::WINDOW_SIZE> WINDOW = make_window();

#ifdef IMPL_OPENCL
    /** @brief A suitable number for local workgroups. */
    static constexpr usize NUM_WORKERS = 256;

    /** @brief The source code of the OpenCL kernels. */
    static constexpr char KERNEL_SRC[] = {
        #embed "kernels.cl"
        , '\0'
    };

    std::shared_ptr<GpuContext> GpuContext::create(usize deviceIdx) {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        usize idx = 0;
        for (const cl::Platform& platform : platforms) {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for (const cl::Device& device : devices) {
                if (idx++ == deviceIdx) {
                    auto gpuContext = std::make_shared<GpuContext>();
                    gpuContext->device = device;
                    gpuContext->rawDevice = gpuContext->device.get();
                    gpuContext->context = cl::Context(gpuContext->device);
                    gpuContext->rawContext = gpuContext->context.get();
                    gpuContext->queue = cl::CommandQueue(
                        gpuContext->context,
                        gpuContext->device
                    );
                    gpuContext->rawQueue = gpuContext->queue.get();
                    cl::Program program(gpuContext->context, KERNEL_SRC);
                    program.build(
                        std::format(
                            "-DWINDOW_SIZE={} -DHOP_SIZE={} -DNUM_BINS={}",
                            Stft::WINDOW_SIZE,
                            Stft::HOP_SIZE,
                            Stft::NUM_BINS
                        )
                    );
                    gpuContext->applyWindow = cl::Kernel(
                        program,
                        "apply_window"
                    );
                    gpuContext->applyGains = cl::Kernel(program, "apply_gains");
                    gpuContext->overlapAdd = cl::Kernel(program, "overlap_add");
                    gpuContext->windowBuffer = cl::Buffer(
                        gpuContext->context,
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        Stft::WINDOW_SIZE * sizeof(f32),
                        const_cast<f32*>(WINDOW.data())
                    );
                    gpuContext->numFwdBatches = 0;
                    gpuContext->numInvBatches = 0;
                    return gpuContext;
                }
            }
        }
        throw std::runtime_error(
            std::format("No OpenCL device found with index '{}'.", deviceIdx)
        );
    }

    /**
     * @brief Rounds up a value to the nearest multiple of another value.
     *
     * @tparam T The type of the value to round up.
     * @tparam U The type of the multiple to round up to.
     * @param[in] value    The value to round up.
     * @param[in] multiple The multiple to round up to.
     * @return The value rounded up to the nearest multiple of the specified
     * multiple.
     */
    template<std::unsigned_integral T, std::unsigned_integral U>
    static constexpr auto round_up(T value, U multiple) noexcept {
        return (value + multiple - 1) / multiple * multiple;
    }
#endif

Stft::Stft(
    AudioFormat format,
    u32 numFrames,
    std::unique_ptr<std::complex<f32>[]> bins
#ifdef IMPL_OPENCL
    , std::shared_ptr<GpuContext> gpuContext,
    cl::Buffer gpuBins
#endif
)
    : _format(format),
      _numFrames(numFrames),
      _bins(std::move(bins))
#ifdef IMPL_OPENCL
      , _gpuContext(gpuContext),
      _gpuBins(gpuBins)
#endif
    { }

#ifdef IMPL_OPENCL
    void Stft::ensure_bins_fetched() {
        if (_bins == nullptr) {
            usize numBatches = static_cast<usize>(_format.numChannels)
                * _numFrames;
            _bins = std::make_unique_for_overwrite<std::complex<f32>[]>(
                numBatches * NUM_BINS
            );
            _gpuContext->queue.enqueueReadBuffer(
                _gpuBins,
                CL_TRUE,
                0,
                numBatches * NUM_BINS * sizeof(std::complex<f32>),
                _bins.get()
            );
        }
    }
#endif

#ifdef IMPL_CUSTOM
    /**
     * @brief Returns an array with the twiddle factors for an FFT of size
     * `WINDOW_SIZE`.
     *
     * @param[in] inverse Whether to compute the twiddle factors for the forward
     *                    or inverse FFT.
     * @return An array with the twiddle factors for an FFT of size
     * `WINDOW_SIZE`.
     */
    static constexpr inline std::array<std::complex<f32>, Stft::HALF_WINDOW_SIZE>
        make_twiddles(bool inverse = false) {
        std::array<std::complex<f32>, Stft::HALF_WINDOW_SIZE> twiddles;
        for (usize k = 0; k < Stft::HALF_WINDOW_SIZE; k++) {
            f32 angle = (inverse ? 2.0F : -2.0F) * PI * static_cast<f32>(k)
                / static_cast<f32>(Stft::WINDOW_SIZE);
            twiddles[k] = std::complex<f32>(std::cos(angle), std::sin(angle));
        }
        return twiddles;
    }

    /** @brief Precomputed twiddle factors for the forward FFT. */
    static constexpr std::array<std::complex<f32>, Stft::HALF_WINDOW_SIZE>
        FORWARD_TWIDDLES = make_twiddles();

    /** @brief Precomputed twiddle factors for the inverse FFT. */
    static constexpr std::array<std::complex<f32>, Stft::HALF_WINDOW_SIZE>
        INVERSE_TWIDDLES = make_twiddles(true);

    /**
     * @brief Returns an array with the bit-reversal permutation indices for an
     * FFT of size `HALF_WINDOW_SIZE`.
     *
     * @return An array with the bit-reversal permutation indices for an FFT of
     * size `HALF_WINDOW_SIZE`.
     */
    static constexpr inline std::array<usize, Stft::HALF_WINDOW_SIZE>
        make_bit_reversal() {
        std::array<usize, Stft::HALF_WINDOW_SIZE> indices;
        for (usize i = 0; i < Stft::HALF_WINDOW_SIZE; i++) {
            usize j = 0;
            for (usize k = 0; k < Stft::HALF_WINDOW_SIZE_BITS; k++) {
                j = (j << 1) | ((i >> k) & 1);
            }
            indices[i] = j;
        }
        return indices;
    }

    /** @brief Precomputed bit-reversal permutation indices. */
    static constexpr std::array<usize, Stft::HALF_WINDOW_SIZE> BIT_REVERSAL =
        make_bit_reversal();

    /**
     * @brief Computes the in-place complex-to-complex (C2C) Fast Fourier
     * Transform (FFT) of a specified buffer of complex numbers.
     *
     * @param[in, out] buffer   The buffer of complex numbers to transform.
     * @param[in]      twiddles A buffer of precomputed twiddle factors for the
     *                          FFT.
     * @param[in]      inverse  Whether to compute the forward or inverse FFT.
     */
    static inline void fft(
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
        for (usize s = 1; s <= Stft::HALF_WINDOW_SIZE_BITS; s++) {
            usize m = 1ULL << s;
            usize stride = Stft::HALF_WINDOW_SIZE >> (s - 1);
            for (usize k = 0; k < Stft::HALF_WINDOW_SIZE; k += m) {
                for (usize j = 0; j < (m / 2); j++) {
                    std::complex<f32> w = twiddles[j * stride];
                    std::complex<f32> t = w * buffer[k + j + m / 2];
                    std::complex<f32> u = buffer[k + j];
                    buffer[k + j] = u + t;
                    buffer[k + j + m / 2] = u - t;
                }
            }
        }
        if (inverse) {
            for (std::complex<f32>& x : buffer) {
                x /= static_cast<f32>(Stft::HALF_WINDOW_SIZE);
            }
        }
    }
#endif

Stft Stft::forward(
    AudioFormat format,
    std::span<const f32> samples
#ifdef IMPL_OPENCL
    , std::shared_ptr<GpuContext> gpuContext
#endif
) {
    usize numFrames = (format.numFrames + HOP_SIZE - 1) / HOP_SIZE;
    usize numBatches = static_cast<usize>(format.numChannels) * numFrames;
#if defined(IMPL_CUSTOM) || defined(IMPL_FFTW)
    auto bins = std::make_unique_for_overwrite<std::complex<f32>[]>(
        numBatches * NUM_BINS
    );
#elifdef IMPL_OPENCL
    std::unique_ptr<std::complex<f32>[]> bins;
#endif
#ifdef IMPL_CUSTOM
    std::array<std::complex<f32>, HALF_WINDOW_SIZE> buffer;
    for (usize c = 0; c < format.numChannels; c++) {
        for (usize f = 0; f < numFrames; f++) {
            for (usize k = 0; k < HALF_WINDOW_SIZE; k++) {
                usize idx0 = f * HOP_SIZE + 2 * k;
                usize idx1 = idx0 + 1;
                f32 x0 = (idx0 < format.numFrames)
                    ? samples[c * format.numFrames + idx0] * WINDOW[2 * k]
                    : 0.0F;
                f32 x1 = (idx1 < format.numFrames)
                    ? samples[c * format.numFrames + idx1] * WINDOW[2 * k + 1]
                    : 0.0F;
                buffer[k] = std::complex<f32>(x0, x1);
            }
            fft(buffer, FORWARD_TWIDDLES);
            std::span<std::complex<f32>> dst(
                bins.get() + (c * numFrames + f) * NUM_BINS,
                NUM_BINS
            );
            dst[0] = buffer[0].real() + buffer[0].imag();
            for (usize k = 1; k < HALF_WINDOW_SIZE; k++) {
                std::complex<f32> zk = buffer[k];
                std::complex<f32> zkc = std::conj(buffer[HALF_WINDOW_SIZE - k]);
                dst[k] = 0.5F * (zk + zkc)
                    - 0.5F * I * FORWARD_TWIDDLES[k] * (zk - zkc);
            }
            dst[HALF_WINDOW_SIZE] = buffer[0].real() - buffer[0].imag();
        }
    }
#elifdef IMPL_FFTW
    std::unique_ptr<f32[], decltype(&fftwf_free)> in(
        fftwf_alloc_real(WINDOW_SIZE),
        &fftwf_free
    );
    std::unique_ptr<std::complex<f32>[], decltype(&fftwf_free)> out(
        reinterpret_cast<std::complex<f32>*>(fftwf_alloc_complex(NUM_BINS)),
        &fftwf_free
    );
    std::unique_ptr<
        std::remove_pointer_t<fftwf_plan>,
        decltype(&fftwf_destroy_plan)
    > plan(
        fftwf_plan_dft_r2c_1d(
            static_cast<int>(WINDOW_SIZE),
            in.get(),
            reinterpret_cast<fftwf_complex*>(out.get()),
            FFTW_MEASURE
        ),
        &fftwf_destroy_plan
    );
    for (usize c = 0; c < format.numChannels; c++) {
        for (usize f = 0; f < numFrames; f++) {
            for (usize k = 0; k < WINDOW_SIZE; k++) {
                usize idx = f * HOP_SIZE + k;
                in[k] = (idx < format.numFrames)
                    ? samples[c * format.numFrames + idx] * WINDOW[k]
                    : 0.0F;
            }
            fftwf_execute(plan.get());
            std::ranges::copy_n(
                out.get(),
                NUM_BINS,
                bins.get() + (c * numFrames + f) * NUM_BINS
            );
        }
    }
#elifdef IMPL_OPENCL
    cl::Buffer samplesBuffer(
        gpuContext->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        format.numChannels * format.numFrames * sizeof(f32),
        const_cast<f32*>(samples.data())
    );
    usize inputBufferSize = numBatches * WINDOW_SIZE * sizeof(f32);
    cl::Buffer inputBuffer(
        gpuContext->context,
        CL_MEM_READ_WRITE,
        inputBufferSize
    );
    cl_mem rawInputBuffer = inputBuffer.get();
    gpuContext->applyWindow.setArg(0, inputBuffer);
    gpuContext->applyWindow.setArg(1, samplesBuffer);
    gpuContext->applyWindow.setArg(2, gpuContext->windowBuffer);
    gpuContext->applyWindow.setArg(3, static_cast<cl_ulong>(numFrames));
    gpuContext->applyWindow.setArg(4, static_cast<cl_ulong>(format.numFrames));
    gpuContext->queue.enqueueNDRangeKernel(
        gpuContext->applyWindow,
        cl::NullRange,
        cl::NDRange(numBatches, round_up(WINDOW_SIZE, NUM_WORKERS)),
        cl::NDRange(1, NUM_WORKERS)
    );
    usize outputBufferSize = numBatches * NUM_BINS * sizeof(std::complex<f32>);
    cl::Buffer outputBuffer(
        gpuContext->context,
        CL_MEM_READ_WRITE,
        outputBufferSize
    );
    cl_mem rawOutputBuffer = outputBuffer.get();
    if (numBatches != gpuContext->numFwdBatches) {
        VkFFTConfiguration config = { };
        config.makeForwardPlanOnly = 1;
        config.FFTdim = 2;
        config.size[0] = WINDOW_SIZE;
        config.size[1] = numBatches;
        config.omitDimension[1] = 1;
        config.performR2C = 1;
        config.device = &gpuContext->rawDevice;
        config.context = &gpuContext->rawContext;
        config.isInputFormatted = 1;
        config.inputBuffer = &rawInputBuffer;
        config.inputBufferSize = &inputBufferSize;
        config.buffer = &rawOutputBuffer;
        config.bufferSize = &outputBufferSize;
        gpuContext->numFwdBatches = numBatches;
        gpuContext->fwdApp.reset(new VkFFTApplication());
        VkFFTResult result = initializeVkFFT(gpuContext->fwdApp.get(), config);
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
    params.inputBuffer = &rawInputBuffer;
    params.buffer = &rawOutputBuffer;
    VkFFTResult result = VkFFTAppend(gpuContext->fwdApp.get(), -1, &params);
    if (result != VKFFT_SUCCESS) {
        throw std::runtime_error(
            std::format(
                "Failed to execute VkFFT: {}.",
                getVkFFTErrorString(result)
            )
        );
    }
    gpuContext->queue.finish();
#endif
    return Stft(
        format,
        static_cast<u32>(numFrames),
        std::move(bins)
    #ifdef IMPL_OPENCL
        , gpuContext,
        outputBuffer
    #endif
    );
}

const AudioFormat& Stft::format() const noexcept {
    return _format;
}

u32 Stft::num_frames() const noexcept {
    return _numFrames;
}

/**
 * @brief Computes the gain of a lowpass filter for a specified frequency and
 * cutoff frequency.
 *
 * @param[in] freq   The frequency to compute the gain for (in Hz).
 * @param[in] cutoff The cutoff frequency of the lowpass filter (in Hz).
 * @return The gain of the lowpass filter for the specified frequency and cutoff
 * frequency.
 */
static inline f32 lowpass_gain(f32 freq, f32 cutoff) noexcept {
    f32 bandwidth = cutoff * ROLLOFF_FRAC;
    f32 low = cutoff - bandwidth;
    f32 high = cutoff + bandwidth;
    if (freq <= low) {
        return 1.0F;
    }
    else if (freq >= high) {
        return 0.0F;
    }
    return 0.5F * (1.0F + std::cos(PI * (freq - low) / (2.0F * bandwidth)));
}

/**
 * @brief Computes the gain of a highpass filter for a specified frequency and
 * cutoff frequency.
 *
 * @param[in] freq   The frequency to compute the gain for (in Hz).
 * @param[in] cutoff The cutoff frequency of the highpass filter (in Hz).
 * @return The gain of the highpass filter for the specified frequency and
 * cutoff frequency.
 */
static inline f32 highpass_gain(f32 freq, f32 cutoff) noexcept {
    return 1.0F - lowpass_gain(freq, cutoff);
}

/**
 * @brief Computes the gain of a bandpass filter for a specified frequency and
 * low and high cutoff frequencies.
 *
 * @param[in] freq       The frequency to compute the gain for (in Hz).
 * @param[in] lowCutoff  The lower cutoff frequency of the bandpass filter (in
 *                       Hz).
 * @param[in] highCutoff The upper cutoff frequency of the bandpass filter (in
 *                       Hz).
 * @return The gain of the bandpass filter for the specified frequency and low
 * and high cutoff frequencies.
 */
static inline f32 bandpass_gain(
    f32 freq,
    f32 lowCutoff,
    f32 highCutoff
) noexcept {
    f32 lowBandwidth = lowCutoff * ROLLOFF_FRAC;
    f32 highBandwidth = highCutoff * ROLLOFF_FRAC;
    f32 riseStart = lowCutoff - lowBandwidth;
    f32 riseEnd = lowCutoff + lowBandwidth;
    f32 fallStart = highCutoff - highBandwidth;
    f32 fallEnd = highCutoff + highBandwidth;
    if ((freq < riseStart) || (freq > fallEnd)) {
        return 0.0F;
    }
    if (freq <= riseEnd) {
        return 0.5F
            * (1.0F
               - std::cos(PI * (freq - riseStart) / (2.0F * lowBandwidth)));
    }
    if (freq >= fallStart) {
        return 0.5F
            * (1.0F
               + std::cos(PI * (freq - fallStart) / (2.0F * highBandwidth)));
    }
    return 1.0F;
}

/**
 * @brief Computes the gain of a bandstop filter for a specified frequency and
 * low and high cutoff frequencies.
 *
 * @param[in] freq       The frequency to compute the gain for (in Hz).
 * @param[in] lowCutoff  The lower cutoff frequency of the bandstop filter (in
 *                       Hz).
 * @param[in] highCutoff The upper cutoff frequency of the bandstop filter (in
 *                       Hz).
 * @return The gain of the bandstop filter for the specified frequency and low
 * and high cutoff frequencies.
 */
static inline f32 bandstop_gain(
    f32 freq,
    f32 lowCutoff,
    f32 highCutoff
) noexcept {
    return 1.0F - bandpass_gain(freq, lowCutoff, highCutoff);
}

void Stft::apply(std::span<const Filter> filters) {
    if (filters.empty()) {
        return;
    }
    std::array<f32, NUM_BINS> gains;
    std::ranges::fill_n(gains.begin(), NUM_BINS, 1.0F);
    for (const Filter& filter : filters) {
        std::visit(
            [&](const auto& f) {
                using T = std::decay_t<decltype(f)>;
                for (usize k = 0; k < NUM_BINS; k++) {
                    f32 freq = static_cast<f32>(k)
                        * static_cast<f32>(_format.sampleRate)
                            / static_cast<f32>(WINDOW_SIZE);
                    if constexpr (std::is_same_v<T, LowpassFilter>) {
                        gains[k] *= lowpass_gain(freq, f.cutoff);
                    }
                    else if constexpr (std::is_same_v<T, HighpassFilter>) {
                        gains[k] *= highpass_gain(freq, f.cutoff);
                    }
                    else if constexpr (std::is_same_v<T, BandpassFilter>) {
                        gains[k] *= bandpass_gain(
                            freq,
                            f.lowCutoff,
                            f.highCutoff
                        );
                    }
                    else {
                        gains[k] *= bandstop_gain(
                            freq,
                            f.lowCutoff,
                            f.highCutoff
                        );
                    }
                }
            },
            filter
        );
    }
    usize numBatches = static_cast<usize>(_format.numChannels) * _numFrames;
#if defined(IMPL_CUSTOM) || defined(IMPL_FFTW)
    for (usize i = 0; i < numBatches; i++) {
        std::span<std::complex<f32>> frame(
            _bins.get() + i * NUM_BINS,
            NUM_BINS
        );
        for (usize k = 0; k < NUM_BINS; k++) {
            frame[k] *= gains[k];
        }
    }
#elifdef IMPL_OPENCL
    cl::Buffer gainsBuffer(
        _gpuContext->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        NUM_BINS * sizeof(f32),
        gains.data()
    );
    _gpuContext->applyGains.setArg(0, _gpuBins);
    _gpuContext->applyGains.setArg(1, gainsBuffer);
    _gpuContext->queue.enqueueNDRangeKernel(
        _gpuContext->applyGains,
        cl::NullRange,
        cl::NDRange(numBatches, round_up(NUM_BINS, NUM_WORKERS)),
        cl::NDRange(1, NUM_WORKERS)
    );
    _gpuContext->queue.finish();
    _bins = nullptr;
#endif
}

std::span<const std::complex<f32>> Stft::bins() {
#ifdef IMPL_OPENCL
    ensure_bins_fetched();
#endif
    return std::span<const std::complex<f32>>(
        _bins.get(),
        _format.numChannels * _numFrames * NUM_BINS
    );
}

std::pair<AudioFormat, std::unique_ptr<f32[]>> Stft::inverse() const {
    auto samples = std::make_unique<f32[]>(
        static_cast<usize>(_format.numChannels) * _format.numFrames
    );
    usize numBatches = static_cast<usize>(_format.numChannels) * _numFrames;
#if defined(IMPL_CUSTOM) || defined(IMPL_FFTW)
    std::unique_ptr<f32[]> outputBuffer = std::make_unique_for_overwrite<f32[]>(
        numBatches * WINDOW_SIZE
    );
#endif
#ifdef IMPL_CUSTOM
    std::array<std::complex<f32>, HALF_WINDOW_SIZE> inputBuffer;
    for (usize c = 0; c < _format.numChannels; c++) {
        for (usize f = 0; f < _numFrames; f++) {
            std::span<std::complex<f32>> src(
                _bins.get() + (c * _numFrames + f) * NUM_BINS,
                NUM_BINS
            );
            for (usize k = 0; k < HALF_WINDOW_SIZE; k++) {
                std::complex<f32> xk = src[k];
                std::complex<f32> xkc = std::conj(src[HALF_WINDOW_SIZE - k]);
                inputBuffer[k] = 0.5F * (xk + xkc)
                    + 0.5F * I * INVERSE_TWIDDLES[k] * (xk - xkc);
            }
            fft(inputBuffer, INVERSE_TWIDDLES, true);
            for (usize k = 0; k < HALF_WINDOW_SIZE; k++) {
                outputBuffer[(c * _numFrames + f) * WINDOW_SIZE + 2 * k] =
                    inputBuffer[k].real() * WINDOW[2 * k];
                outputBuffer[(c * _numFrames + f) * WINDOW_SIZE + 2 * k + 1] =
                    inputBuffer[k].imag() * WINDOW[2 * k + 1];
            }
        }
    }
#elifdef IMPL_FFTW
    std::unique_ptr<std::complex<f32>[], decltype(&fftwf_free)> in(
        reinterpret_cast<std::complex<f32>*>(fftwf_alloc_complex(NUM_BINS)),
        &fftwf_free
    );
    std::unique_ptr<f32[], decltype(&fftwf_free)> out(
        fftwf_alloc_real(WINDOW_SIZE),
        &fftwf_free
    );
    std::unique_ptr<
        std::remove_pointer_t<fftwf_plan>,
        decltype(&fftwf_destroy_plan)
    > plan(
        fftwf_plan_dft_c2r_1d(
            static_cast<int>(WINDOW_SIZE),
            reinterpret_cast<fftwf_complex*>(in.get()),
            out.get(),
            FFTW_MEASURE
        ),
        &fftwf_destroy_plan
    );
    for (usize c = 0; c < _format.numChannels; c++) {
        for (usize f = 0; f < _numFrames; f++) {
            std::span<const std::complex<f32>> src(
                _bins.get() + (c * _numFrames + f) * NUM_BINS,
                NUM_BINS
            );
            std::ranges::copy_n(src.begin(), NUM_BINS, in.get());
            fftwf_execute(plan.get());
            for (usize k = 0; k < WINDOW_SIZE; k++) {
                outputBuffer[(c * _numFrames + f) * WINDOW_SIZE + k] =
                    out[k] / static_cast<f32>(WINDOW_SIZE) * WINDOW[k];
            }
        }
    }
#elifdef IMPL_OPENCL
    usize inputBufferSize = numBatches * NUM_BINS * sizeof(std::complex<f32>);
    cl_mem rawInputBuffer = _gpuBins.get();
    usize outputBufferSize = numBatches * WINDOW_SIZE * sizeof(f32);
    cl::Buffer outputBuffer(
        _gpuContext->context,
        CL_MEM_READ_WRITE,
        outputBufferSize
    );
    cl_mem rawOutputBuffer = outputBuffer.get();
    if (numBatches != _gpuContext->numInvBatches) {
        VkFFTConfiguration config = { };
        config.makeInversePlanOnly = 1;
        config.FFTdim = 2;
        config.size[0] = WINDOW_SIZE;
        config.size[1] = numBatches;
        config.omitDimension[1] = 1;
        config.performR2C = 1;
        config.device = &_gpuContext->rawDevice;
        config.context = &_gpuContext->rawContext;
        config.buffer = &rawInputBuffer;
        config.bufferSize = &inputBufferSize;
        config.isOutputFormatted = 1;
        config.outputBuffer = &rawOutputBuffer;
        config.outputBufferSize = &outputBufferSize;
        _gpuContext->numInvBatches = numBatches;
        _gpuContext->invApp.reset(new VkFFTApplication());
        VkFFTResult result = initializeVkFFT(_gpuContext->invApp.get(), config);
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
    params.buffer = &rawInputBuffer;
    params.outputBuffer = &rawOutputBuffer;
    VkFFTResult result = VkFFTAppend(_gpuContext->invApp.get(), 1, &params);
    if (result != VKFFT_SUCCESS) {
        throw std::runtime_error(
            std::format(
                "Failed to execute VkFFT: {}.",
                getVkFFTErrorString(result)
            )
        );
    }
    usize samplesBufferSize = _format.numChannels * _format.numFrames
        * sizeof(f32);
    cl::Buffer samplesBuffer(
        _gpuContext->context,
        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        samplesBufferSize
    );
    _gpuContext->overlapAdd.setArg(0, samplesBuffer);
    _gpuContext->overlapAdd.setArg(1, outputBuffer);
    _gpuContext->overlapAdd.setArg(2, _gpuContext->windowBuffer);
    _gpuContext->overlapAdd.setArg(3, static_cast<cl_ulong>(_numFrames));
    _gpuContext->overlapAdd.setArg(4, static_cast<cl_ulong>(_format.numFrames));
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
        samplesBuffer,
        CL_TRUE,
        0,
        samplesBufferSize,
        samples.get()
    );
#endif
#if defined(IMPL_CUSTOM) || defined(IMPL_FFTW)
    for (usize c = 0; c < _format.numChannels; c++) {
        for (usize idx = 0; idx < _format.numFrames; idx++) {
            f32 sum = 0.0F;
            f32 norm = 0.0F;
            usize first = (idx >= WINDOW_SIZE)
                ? (idx - WINDOW_SIZE) / HOP_SIZE + 1
                : 0;
            for (usize f = first; f < _numFrames; f++) {
                usize k = idx - f * HOP_SIZE;
                if (k >= WINDOW_SIZE) {
                    break;
                }
                sum += outputBuffer[(c * _numFrames + f) * WINDOW_SIZE + k];
                norm += WINDOW[k] * WINDOW[k];
            }
            samples[c * _format.numFrames + idx] = (norm > 1.0E-8F)
                ? sum / norm
                : 0.0F;
        }
    }
#endif
    return std::make_pair(_format, std::move(samples));
}

}