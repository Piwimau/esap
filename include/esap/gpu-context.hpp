#ifndef ESAP_GPU_CONTEXT_HPP
#define ESAP_GPU_CONTEXT_HPP

#include <memory>
#include <optional>
#include <string>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#define VKFFT_BACKEND 3
#include <vkFFT.h>

#include "esap/func-deleter.hpp"
#include "esap/types.hpp"

namespace esap {

/**
 * @brief Deletes a `VkFFTApplication` handle.
 *
 * @warning The behavior is undefined if `app` has already been deleted. Also
 * note that this function assumes that `app` was allocated by `operator new`.
 *
 * @param[in, out] app The `VkFFTApplication` handle to delete.
 */
inline void delete_vkfft_application(VkFFTApplication* app) noexcept {
    if (app != nullptr) {
        deleteVkFFT(app);
        delete app;
    }
}

/** @brief Represents a custom deleter for `VkFFTApplication` handles. */
using VkFFTApplicationDeleter = FuncDeleter<&delete_vkfft_application>;

/** @brief Represents a context for GPU-accelerated STFT computations. */
struct GpuContext {

    /** @brief The device associated with this GPU context. */
    cl::Device device;

    /** @brief The raw device, required for interfacing with VkFFT. */
    cl_device_id rawDevice;

    /** @brief The underlying context associated with this GPU context. */
    cl::Context context;

    /**
     * @brief The raw underlying context, required for interfacing with VkFFT.
     */
    cl_context rawContext;

    /** @brief The command queue associated with this GPU context. */
    cl::CommandQueue queue;

    /** @brief The raw command queue, required for interfacing with VkFFT. */
    cl_command_queue rawQueue;

    /**
     * @brief The kernel for applying a window function to the time-domain
     * samples.
     */
    cl::Kernel applyWindow;

    /** @brief The kernel for applying gains to the complex frequency bins. */
    cl::Kernel applyGains;

    /** @brief The kernel for reconstructing the time-domain signal. */
    cl::Kernel overlapAdd;

    /** @brief The precomputed coefficients of the window function. */
    cl::Buffer windowBuffer;

    /** @brief The `VkFFTApplication` for the forward FFT. */
    std::unique_ptr<VkFFTApplication, VkFFTApplicationDeleter> fwdApp;

    /** @brief The `VkFFTApplication` for the inverse FFT. */
    std::unique_ptr<VkFFTApplication, VkFFTApplicationDeleter> invApp;

    /** @brief The number of FFTs computed in the forward transformation. */
    usize numFwdBatches;

    /** @brief The number of FFTs computed in the inverse transformation. */
    usize numInvBatches;

    /**
     * @brief Creates a GPU context for an OpenCL device matching a specified
     * pattern.
     *
     * @param[in] pattern The pattern used to match the desired OpenCL device
     *                    (as specified by the output of the `--list-devices`
     *                    command-line option).
     * @return A GPU context associated with the specified OpenCL device.
     * @throws `cl::Error`          Thrown when an OpenCL error occurs while
     *                              creating the GPU context.
     * @throws `std::runtime_error` Thrown when no device is found, when no
     *                              device matches the specified pattern, or
     *                              when multiple devices match the specified
     *                              pattern.
     */
    static GpuContext create(std::optional<std::string> pattern);

};

}

#endif