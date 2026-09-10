#include <algorithm>
#include <cctype>
#include <format>
#include <stdexcept>
#include <vector>
#include "esap/gpu-context.hpp"
#include "esap/stft.hpp"

namespace esap {

/** @brief The source code of the OpenCL kernels. */
static constexpr char KERNEL_SRC[] = {
    #embed "kernels.cl"
    , '\0'
};

/** @brief The precomputed coefficients of a Hann window. */
static constexpr std::array<f32, Stft::WINDOW_SIZE> WINDOW = make_window();

/**
 * @brief Selects the first available OpenCL device.
 *
 * @return The first available OpenCL device.
 * @throws `std::runtime_error` Thrown when no OpenCL devices are found.
 */
static cl::Device select_first_device() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (const cl::Platform& platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (!devices.empty()) {
            return devices.front();
        }
    }
    throw std::runtime_error("No OpenCL devices found.");
}

/**
 * @brief Selects an OpenCL device matching a specified pattern.
 *
 * @param[in] pattern The pattern used to match the desired OpenCL device
 *                    (as specified by the output of the `--list-devices`
 *                    command-line option).
 * @return The OpenCL device that matches the specified pattern.
 * @throws `std::runtime_error` Thrown when no devices match the specified
 *                              pattern, or when multiple devices match the
 *                              pattern.
 */
static cl::Device select_device_by_pattern(const std::string& pattern) {
    std::string needle = pattern;
    std::ranges::transform(
        needle,
        needle.begin(),
        [](unsigned char c) { return std::tolower(c); }
    );
    std::vector<cl::Device> candidates;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (const cl::Platform& platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (const cl::Device& device : devices) {
            std::string haystack = device.getInfo<CL_DEVICE_NAME>();
            std::ranges::transform(
                haystack,
                haystack.begin(),
                [](unsigned char c) { return std::tolower(c); }
            );
            if (haystack.contains(needle)) {
                candidates.push_back(device);
            }
        }
    }
    if (candidates.empty()) {
        throw std::runtime_error(
            std::format("No OpenCL devices match the pattern '{}'.", pattern)
        );
    }
    if (std::ssize(candidates) > 1) {
        throw std::runtime_error(
            std::format(
                "Multiple OpenCL devices match the pattern '{}'.",
                pattern
            )
        );
    }
    return candidates.front();
}

GpuContext GpuContext::create(std::optional<std::string> pattern) {
    cl::Device device = pattern.has_value()
        ? select_device_by_pattern(pattern.value())
        : select_first_device();
    GpuContext gpuContext;
    gpuContext.device = device;
    gpuContext.rawDevice = gpuContext.device.get();
    gpuContext.context = cl::Context(gpuContext.device);
    gpuContext.rawContext = gpuContext.context.get();
    gpuContext.queue = cl::CommandQueue(gpuContext.context, gpuContext.device);
    gpuContext.rawQueue = gpuContext.queue.get();
    cl::Program program(gpuContext.context, KERNEL_SRC);
    program.build(
        std::format(
            "-DWINDOW_SIZE={} -DHOP_SIZE={} -DNUM_BINS={}",
            Stft::WINDOW_SIZE,
            Stft::HOP_SIZE,
            Stft::NUM_BINS
        )
    );
    gpuContext.applyWindow = cl::Kernel(program, "apply_window");
    gpuContext.applyGains = cl::Kernel(program, "apply_gains");
    gpuContext.overlapAdd = cl::Kernel(program, "overlap_add");
    gpuContext.windowBuffer = cl::Buffer(
        gpuContext.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        Stft::WINDOW_SIZE * sizeof(f32),
        const_cast<f32*>(WINDOW.data())
    );
    gpuContext.numFwdBatches = 0;
    gpuContext.numInvBatches = 0;
    return gpuContext;
}

}