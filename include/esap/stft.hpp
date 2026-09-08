#ifndef ESAP_STFT_HPP
#define ESAP_STFT_HPP

#include <bit>
#include <complex>
#include <memory>
#include <span>
#include <utility>
#ifdef IMPL_OPENCL
    #define CL_HPP_TARGET_OPENCL_VERSION 300
    #define CL_HPP_ENABLE_EXCEPTIONS
    #include <CL/opencl.hpp>

    #define VKFFT_BACKEND 3
    #include <vkFFT.h>
#endif
#include "esap/audio-format.hpp"
#include "esap/filter.hpp"
#include "esap/types.hpp"

namespace esap {

#ifdef IMPL_OPENCL
    /** @brief Represents a custom deleter for `VkFFTApplication` handles. */
    struct VkFFTApplicationDeleter {

        /**
         * @brief Deletes a specified `VkFFTApplication`.
         *
         * @warning The behavior is undefined if `app` is a `nullptr` or if it
         * has already been deleted. Note that this function also assumes that
         * `app` was allocated by operator `new`.
         *
         * @param[in, out] app The `VkFFTApplication` to delete.
         */
        void operator()(VkFFTApplication* app) const noexcept {
            deleteVkFFT(app);
            delete app;
        }

    };

    /** @brief Represents a GPU context for OpenCL-based STFT computations. */
    struct GpuContext {

        /** @brief The GPU device to be used for computations. */
        cl::Device device;

        /**
         * @brief The raw OpenCL device, which is required for interfacing with
         * the VkFFT library.
         */
        cl_device_id rawDevice;

        /** @brief The OpenCL context to be used for computations. */
        cl::Context context;

        /**
         * @brief The raw OpenCL context, which is required for interfacing with
         * the VkFFT library.
         */
        cl_context rawContext;

        /** @brief The OpenCL command queue to be used for computations. */
        cl::CommandQueue queue;

        /**
         * @brief The raw OpenCL command queue, which is required for
         * interfacing with the VkFFT library.
         */
        cl_command_queue rawQueue;

        /** @brief The OpenCL kernel for applying a window to the samples. */
        cl::Kernel applyWindow;

        /** @brief The OpenCL kernel for applying gains to the bins. */
        cl::Kernel applyGains;

        /** @brief The OpenCL kernel for performing the overlap-add method. */
        cl::Kernel overlapAdd;

        /** @brief The precomputed coefficients of the window function. */
        cl::Buffer windowBuffer;

        /** @brief The VkFFT application for forward FFT computations. */
        std::unique_ptr<VkFFTApplication, VkFFTApplicationDeleter> fwdApp;

        /** @brief The VkFFT application for inverse FFT computations. */
        std::unique_ptr<VkFFTApplication, VkFFTApplicationDeleter> invApp;

        /**
         * @brief The number of batches (i.e., `numChannels * numFrames`) to be
         * processed in each forward FFT computation.
         */
        usize numFwdBatches;

        /**
         * @brief The number of batches (i.e., `numChannels * numFrames`) to be
         * processed in each inverse FFT computation.
         */
        usize numInvBatches;

        /**
         * @brief Creates a new GPU context for OpenCL-based STFT computations.
         *
         * @param[in] deviceIdx The index of the OpenCL device to use (as listed
         *                      by the `--list-devices` command-line option),
         *                      zero by default.
         * @return A new GPU context for OpenCL-based STFT computations.
         * @throws `cl::Error`          Thrown when an OpenCL error occurs
         *                              during the creation of the GPU context.
         * @throws `std::runtime_error` Thrown when no OpenCL device is found
         *                              with the specified index.
         */
        static std::shared_ptr<GpuContext> create(usize deviceIdx = 0);

    };
#endif

/**
 * @brief Represents a Short-Time Fourier Transform (STFT) of an audio signal.
 */
class Stft final {
private:

    /** @brief The format of the original audio signal. */
    AudioFormat _format;

    /**
     * @brief The number of STFT frames (i.e., the number of windowed FFTs
     * computed per audio channel).
     */
    u32 _numFrames;

    /**
     * @brief The bins (i.e., the complex frequency-domain representation of the
     * audio signal).
     *
     * @note The bins are stored as three-dimensional array, where the first
     * dimension corresponds to the audio channel, the second dimension
     * corresponds to the frame, and the third dimension corresponds to the
     * frequency bin (between `0` and `Stft::HalfWindowSize`, inclusive).
     */
    std::unique_ptr<std::complex<f32>[]> _bins;

#ifdef IMPL_OPENCL
    /** @brief The GPU context for OpenCL-based STFT computations. */
    std::shared_ptr<GpuContext> _gpuContext;

    /** @brief The GPU-side buffer for the bins. */
    cl::Buffer _gpuBins;
#endif

    /**
     * @brief Initializes a new Short-Time Fourier Transform (STFT) with the
     * specified parameters.
     *
     * @note This constructor stores a pointer to the GPU context, which is not
     * owned by the `Stft` object, but must remain valid for its lifetime (i.e.,
     * until the `Stft` object is destroyed). Also note that the GPU context may
     * be modified by the `Stft` object.
     *
     * @warning This constructor is a private implementation detail and does not
     * perform any validation on the parameters.
     *
     * @param[in] format     The format of the original audio signal.
     * @param[in] numFrames  The number of audio frames (i.e., the number of
     *                       windowed FFTs computed per audio channel).
     * @param[in] bins       The bins (i.e., the complex frequency-domain
     *                       representation of the audio signal).
     * @param[in] gpuContext The GPU context for OpenCL-based STFT computations
     *                       (required only when `IMPL_OPENCL` is defined).
     * @param[in] gpuBins    The GPU-side buffer for the bins (required only
     *                       when `IMPL_OPENCL` is defined).
     */
    Stft(
        AudioFormat format,
        u32 numFrames,
        std::unique_ptr<std::complex<f32>[]> bins
#ifdef IMPL_OPENCL
        ,
        std::shared_ptr<GpuContext> gpuContext,
        cl::Buffer gpuBins
#endif
    );

#ifdef IMPL_OPENCL
    /**
     * @brief Fetches the bins from the GPU-side buffer if they are not already
     * available on the host.
     *
     * @throws `cl::Error` Thrown when an OpenCL error occurs while fetching the
     *                     bins from the GPU.
     */
    void ensure_bins_fetched();
#endif

public:

    /**
     * @brief The window size (i.e., the number of audio samples in each
     * window).
     */
    static constexpr usize WINDOW_SIZE = 2048;

    /** @brief Half of the window size. */
    static constexpr usize HALF_WINDOW_SIZE = WINDOW_SIZE / 2;

    /**
     * @brief The number of bits required to represent half of the window size.
     */
    static constexpr usize HALF_WINDOW_SIZE_BITS = static_cast<usize>(
        std::countr_zero(HALF_WINDOW_SIZE)
    );

    /**
     * @brief The hop size (i.e., the number of audio samples to advance for
     * each window).
     */
    static constexpr usize HOP_SIZE = HALF_WINDOW_SIZE;

    /** @brief The number of frequency bins in each window. */
    static constexpr usize NUM_BINS = HALF_WINDOW_SIZE + 1;

    /**
     * @brief Computes the Short-Time Fourier Transform (STFT) of an audio
     * signal.
     *
     * @note The resulting `Stft` object stores a pointer to the GPU context,
     * which is not owned by the `Stft` object, but must remain valid for its
     * lifetime. Also note that the GPU context may be modified by the `Stft`
     * object.
     *
     * @param[in] format     The format of the original audio signal.
     * @param[in] samples    The audio samples to compute the STFT of.
     * @param[in] gpuContext The GPU context for OpenCL-based STFT computations
     *                       (required only when `IMPL_OPENCL` is defined).
     * @return An object representing the STFT of the specified audio signal.
     * @throws `std::runtime_error` Thrown when an error occurs during the
     *                              computation of the STFT (only when
     *                              `IMPL_CUSTOM` is defined).
     * @throws `cl::Error`          Thrown when an OpenCL error occurs during
     *                              the computation of the STFT (only when
     *                              `IMPL_OPENCL` is defined).
     */
    static Stft forward(
        AudioFormat format,
        std::span<const f32> samples
#ifdef IMPL_OPENCL
        ,
        std::shared_ptr<GpuContext> gpuContext
#endif
    );

    /**
     * @brief Returns the format of the original audio signal.
     *
     * @return The format of the original audio signal.
     */
    const AudioFormat& format() const noexcept;

    /**
     * @brief Returns the number of STFT frames (i.e., the number of windowed
     * FFTs computed per audio channel).
     *
     * @return The number of STFT frames.
     */
    u32 num_frames() const noexcept;

    /**
     * @brief Applies the specified filters.
     *
     * @param[in] filters The filters to apply.
     */
    void apply(std::span<const Filter> filters);

    /**
     * @brief Returns a read-only view of the bins (i.e., the complex
     * frequency-domain representation of the audio signal).
     *
     * @return A read-only view of the bins.
     * @throws `cl::Error` Thrown when an OpenCL error occurs while fetching the
     *                     bins from the GPU (only when `IMPL_OPENCL` is
     *                     defined).
     */
    std::span<const std::complex<f32>> bins();

    /**
     * @brief Computes the inverse of this Short-Time Fourier Transform (STFT)
     * to reconstruct the original audio signal.
     *
     * @return A pair containing the format of the original audio signal and the
     * reconstructed audio samples.
     * @throws `std::runtime_error` Thrown when an error occurs during the
     *                              computation of the inverse STFT (only when
     *                              `IMPL_CUSTOM` is defined).
     * @throws `cl::Error`          Thrown when an OpenCL error occurs during
     *                              the computation of the inverse STFT (only
     *                              when `IMPL_OPENCL` is defined).
     */
    std::pair<AudioFormat, std::unique_ptr<f32[]>> inverse() const;

};

}

#endif