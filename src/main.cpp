#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <getopt.h>
#include <optional>
#include <print>
#include <string>
#include <utility>
#include <vector>
#include "esap/audio-format.hpp"
#include "esap/bench.hpp"
#include "esap/export.hpp"
#include "esap/filter.hpp"
#include "esap/stft.hpp"
#include "esap/types.hpp"
#include "esap/wav.hpp"

using namespace esap;

/** @brief Represents a phase of the audio pipeline to benchmark. */
enum class Phase {

    /**
     * @brief Indicates that the entire audio pipeline (forward STFT, filtering
     * and inverse STFT) should be benchmarked.
     */
    END_TO_END,

    /** @brief Indicates that only the forward STFT should be benchmarked. */
    FORWARD_ONLY,

    /** @brief Indicates that only the inverse STFT should be benchmarked. */
    INVERSE_ONLY,

    /** @brief Indicates that only the filtering step should be benchmarked. */
    FILTER_ONLY

};

/** @brief Represents a command-line option. */
using Option = struct option;

/** @brief The version information. */
static constexpr std::string VERSION = "0.1.0";

/** @brief The short command-line options. */
static constexpr char SHORT_OPTS[] = ":bf:o:s:S:hv";

/** @brief The identifier for the `--warmups` command-line option. */
static constexpr int OPT_WARMUPS = 256;

/** @brief The identifier for the `--iterations` command-line option. */
static constexpr int OPT_ITERATIONS = 257;

/** @brief The identifier for the `--phase` command-line option. */
static constexpr int OPT_PHASE = 258;

#ifdef IMPL_OPENCL
    /** @brief The identifier for the `--list-devices` command-line option. */
    static constexpr int OPT_LIST_DEVICES = 259;

    /** @brief The identifier for the `--device` command-line option. */
    static constexpr int OPT_DEVICE = 260;
#endif

/** @brief The number of warmup iterations to perform before benchmarking. */
static constexpr usize DEFAULT_WARMUPS = 10;

/** @brief The number of benchmark iterations to perform. */
static constexpr usize DEFAULT_ITERATIONS = 100;

/** @brief The long command-line options. */
static constexpr Option LONG_OPTS[] = {
    { "benchmark", no_argument, nullptr, 'b' },
    { "warmups", required_argument, nullptr, OPT_WARMUPS },
    { "phase", required_argument, nullptr, OPT_PHASE },
    { "iterations", required_argument, nullptr, OPT_ITERATIONS },
    { "filter", required_argument, nullptr, 'f' },
    { "output", required_argument, nullptr, 'o' },
    { "spectrum", required_argument, nullptr, 's' },
    { "spectrogram", required_argument, nullptr, 'S' },
#ifdef IMPL_OPENCL
    { "list-devices", no_argument, nullptr, OPT_LIST_DEVICES },
    { "device", required_argument, nullptr, OPT_DEVICE },
#endif
    { "help", no_argument, nullptr, 'h' },
    { "version", no_argument, nullptr, 'v' },
    { nullptr, 0, nullptr, 0 }
};

template<>
struct std::formatter<Phase> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const Phase& phase, std::format_context& ctx) const {
        std::string_view name;
        switch (phase) {
            case Phase::END_TO_END:
                name = "end-to-end";
                break;
            case Phase::FORWARD_ONLY:
                name = "forward-only";
                break;
            case Phase::INVERSE_ONLY:
                name = "inverse-only";
                break;
            case Phase::FILTER_ONLY:
                name = "filter-only";
                break;
            default:
                std::unreachable();
        }
        return std::format_to(ctx.out(), "{}", name);
    }

};

/** @brief Represents the parsed command-line arguments. */
struct Args {

    /** @brief The name of the program. */
    std::string program;

    /** @brief The path to the input audio file. */
    std::string input;

    /** @brief The path to the output audio file (if specified). */
    std::optional<std::string> output;

    /** @brief The path to the output spectrum CSV file (if specified). */
    std::optional<std::string> spectrum;

    /** @brief The path to the output spectrogram CSV file (if specified). */
    std::optional<std::string> spectrogram;

    /** @brief Whether to benchmark the execution time. */
    bool benchmark;

    /**
     * @brief The number of warmup iterations to perform before benchmarking
     * (if specified).
     */
    std::optional<usize> warmups;

    /** @brief The number of benchmark iterations to perform (if specified). */
    std::optional<usize> iterations;

    /** @brief The phase of the audio pipeline to benchmark (if specified). */
    std::optional<Phase> phase;

    /** @brief The filters to apply (if specified). */
    std::vector<Filter> filters;

#ifdef IMPL_OPENCL
    /** @brief The index of the OpenCL device to use (if specified). */
    std::optional<usize> deviceIdx;
#endif

    /**
     * @brief Parses the command-line arguments.
     *
     * @warning If the list-devices (`--list-devices`), help (`-h` or `--help`)
     * or version (`-v` or `--version`) option is specified, this function will
     * print the corresponding information to the standard output stream and
     * terminate the program with `EXIT_SUCCESS`. If any error occurs during
     * parsing (e.g., missing a required argument, unknown option, etc.), a
     * corresponding error message will be printed to the standard error stream
     * and the program is terminated with `EXIT_FAILURE`.
     *
     * @param[in] argc The number of command-line arguments.
     * @param[in] argv The array of command-line arguments.
     * @return The parsed command-line arguments.
     */
    static Args parse(int argc, char** argv) {
        assert(argc > 0);
        assert(argv != nullptr);
        Args args;
        args.program = std::filesystem::path(argv[0]).filename().string();
        args.benchmark = false;
        int opt;
        while (
            (opt = getopt_long(argc, argv, SHORT_OPTS, LONG_OPTS, nullptr)) != -1
        ) {
            switch (opt) {
                case 'b':
                    args.benchmark = true;
                    break;
                case OPT_WARMUPS: {
                    usize warmups;
                    if (std::sscanf(optarg, "%zu", &warmups) != 1) {
                        std::println(
                            stderr,
                            "{}: Invalid number of warmup iterations '{}'.\n"
                            "Try '{} --help' for more information.",
                            args.program,
                            optarg,
                            args.program
                        );
                        std::exit(EXIT_FAILURE);
                    }
                    args.warmups = warmups;
                    break;
                }
                case OPT_ITERATIONS: {
                    usize iterations;
                    if (std::sscanf(optarg, "%zu", &iterations) != 1) {
                        std::println(
                            stderr,
                            "{}: Invalid number of benchmark iterations '{}'.\n"
                            "Try '{} --help' for more information.",
                            args.program,
                            optarg,
                            args.program
                        );
                        std::exit(EXIT_FAILURE);
                    }
                    args.iterations = iterations;
                    break;
                }
                case OPT_PHASE: {
                    std::string_view phase = optarg;
                    if (phase == "end-to-end") {
                        args.phase = Phase::END_TO_END;
                    }
                    else if (phase == "forward-only") {
                        args.phase = Phase::FORWARD_ONLY;
                    }
                    else if (phase == "inverse-only") {
                        args.phase = Phase::INVERSE_ONLY;
                    }
                    else if (phase == "filter-only") {
                        args.phase = Phase::FILTER_ONLY;
                    }
                    else {
                        std::println(
                            stderr,
                            "{}: Invalid benchmark phase '{}'.\n"
                            "Try '{} --help' for more information.",
                            args.program,
                            phase,
                            args.program
                        );
                        std::exit(EXIT_FAILURE);
                    }
                    break;
                }
                case 'f': {
                    f32 cutoff;
                    f32 lowCutoff;
                    f32 highCutoff;
                    if (std::sscanf(optarg, "lowpass:%f", &cutoff) == 1) {
                        LowpassFilter filter = { .cutoff = cutoff };
                        args.filters.push_back(filter);
                    }
                    else if (std::sscanf(optarg, "highpass:%f", &cutoff) == 1) {
                        HighpassFilter filter = { .cutoff = cutoff };
                        args.filters.push_back(filter);
                    }
                    else if (
                        std::sscanf(
                            optarg,
                            "bandpass:%f:%f",
                            &lowCutoff,
                            &highCutoff
                        ) == 2
                    ) {
                        BandpassFilter filter = {
                            .lowCutoff = lowCutoff,
                            .highCutoff = highCutoff
                        };
                        args.filters.push_back(filter);
                    }
                    else if (
                        std::sscanf(
                            optarg,
                            "bandstop:%f:%f",
                            &lowCutoff,
                            &highCutoff
                        ) == 2
                    ) {
                        BandstopFilter filter = {
                            .lowCutoff = lowCutoff,
                            .highCutoff = highCutoff
                        };
                        args.filters.push_back(filter);
                    }
                    else {
                        std::println(
                            stderr,
                            "{}: Invalid filter specification '{}'.\n"
                            "Try '{} --help' for more information.",
                            args.program,
                            optarg,
                            args.program
                        );
                        std::exit(EXIT_FAILURE);
                    }
                    break;
                }
                case 'o':
                    args.output = optarg;
                    break;
                case 's':
                    args.spectrum = optarg;
                    break;
                case 'S':
                    args.spectrogram = optarg;
                    break;
            #ifdef IMPL_OPENCL
                case OPT_LIST_DEVICES: {
                    std::vector<cl::Platform> platforms;
                    cl::Platform::get(&platforms);
                    usize idx = 0;
                    for (const cl::Platform& platform : platforms) {
                        std::vector<cl::Device> devices;
                        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                        for (const cl::Device& device : devices) {
                            std::println(
                                "[{}] ({}) {}",
                                idx++,
                                platform.getInfo<CL_PLATFORM_NAME>(),
                                device.getInfo<CL_DEVICE_NAME>()
                            );
                        }
                    }
                    std::exit(EXIT_SUCCESS);
                    break;
                }
                case OPT_DEVICE: {
                    usize deviceIdx;
                    if (std::sscanf(optarg, "%zu", &deviceIdx) != 1) {
                        std::println(
                            stderr,
                            "{}: Invalid OpenCL device index '{}'.\n"
                            "Try '{} --help' for more information.",
                            args.program,
                            optarg,
                            args.program
                        );
                        std::exit(EXIT_FAILURE);
                    }
                    args.deviceIdx = deviceIdx;
                    break;
                }
            #endif
                case 'h':
                    std::println(
                        "Usage: {} [OPTION]... FILE\n"
                        "Process the waveform audio file FILE.\n"
                        "\n"
                        "Options:\n"
                        "  -b, --benchmark         Benchmark the execution time (excluding I/O).\n"
                        "      --warmups=N         Perform N warmup iterations before benchmarking (default: {}).\n"
                        "      --iterations=N      Perform N benchmark iterations (default: {}).\n"
                        "      --phase=PHASE       Benchmark only the specified phase of the audio pipeline.\n"
                        "                          PHASE can be one of the following:\n"
                        "                            end-to-end (default)\n"
                        "                            forward-only\n"
                        "                            inverse-only\n"
                        "                            filter-only\n"
                        "  -f, --filter=SPEC       Apply the specified filter.\n"
                        "                          SPEC can be one of the following:\n"
                        "                            lowpass:FREQ\n"
                        "                            highpass:FREQ\n"
                        "                            bandpass:FREQ_LOW:FREQ_HIGH\n"
                        "                            bandstop:FREQ_LOW:FREQ_HIGH\n"
                        "                          FREQ, FREQ_LOW and FREQ_HIGH are cutoff frequencies in Hz.\n"
                        "                          Zero or more filters can be specified, which are applied in the given order.\n"
                        "  -o, --output=FILE       Write the processed audio to FILE.\n"
                        "  -s, --spectrum=FILE     Write the spectrum of the processed audio to FILE in CSV format.\n"
                        "  -S, --spectrogram=FILE  Write the spectrogram of the processed audio to FILE in CSV format.\n"
                    #ifdef IMPL_OPENCL
                        "      --list-devices      List available OpenCL devices and exit.\n"
                        "      --device=IDX        Use the OpenCL device with index IDX (as listed by --list-devices).\n"
                    #endif
                        "  -h, --help              Display this help and exit.\n"
                        "  -v, --version           Display version information and exit.",
                        args.program,
                        DEFAULT_WARMUPS,
                        DEFAULT_ITERATIONS
                    );
                    std::exit(EXIT_SUCCESS);
                    break;
                case 'v':
                    std::println("{} {}", args.program, VERSION);
                    std::exit(EXIT_SUCCESS);
                    break;
                case ':':
                    std::println(
                        stderr,
                        "{}: Option '-{}' requires an argument.\n"
                        "Try '{} --help' for more information.",
                        args.program,
                        static_cast<char>(optopt),
                        args.program
                    );
                    std::exit(EXIT_FAILURE);
                    break;
                case '?':
                    std::println(
                        stderr,
                        "{}: Unknown option '{}'.\n"
                        "Try '{} --help' for more information.",
                        args.program,
                        argv[optind - 1],
                        args.program
                    );
                    std::exit(EXIT_FAILURE);
                    break;
                default:
                    std::unreachable();
            }
        }
        if (optind >= argc) {
            std::println(
                stderr,
                "{}: Missing input file.\n"
                "Try '{} --help' for more information.",
                args.program,
                args.program
            );
            std::exit(EXIT_FAILURE);
        }
        args.input = argv[optind];
        return args;
    }

};

int main(int argc, char** argv) {
    Args args = Args::parse(argc, argv);
    try {
    #ifdef IMPL_OPENCL
        std::shared_ptr<GpuContext> gpuContext = GpuContext::create(
            args.deviceIdx.value_or(0)
        );
    #endif
        Wav input = Wav::read(args.input);
        if (args.benchmark) {
            usize warmups = args.warmups.value_or(DEFAULT_WARMUPS);
            usize iterations = args.iterations.value_or(DEFAULT_ITERATIONS);
            BenchResult result;
            Phase phase = args.phase.value_or(Phase::END_TO_END);
            switch (phase) {
                case Phase::END_TO_END:
                    result = benchmark(
                        warmups,
                        iterations,
                        [&]() {
                            Stft stft = Stft::forward(
                                input.format(),
                                input.samples()
                            #ifdef IMPL_OPENCL
                                , gpuContext
                            #endif
                            );
                            if (!args.filters.empty()) {
                                stft.apply(args.filters);
                            }
                            return stft.inverse();
                        }
                    );
                    break;
                case Phase::FORWARD_ONLY:
                    result = benchmark(
                        warmups,
                        iterations,
                        [&]() {
                            return Stft::forward(
                                input.format(),
                                input.samples()
                            #ifdef IMPL_OPENCL
                                , gpuContext
                            #endif
                            );
                        }
                    );
                    break;
                case Phase::INVERSE_ONLY: {
                    Stft stft = Stft::forward(
                        input.format(),
                        input.samples()
                    #ifdef IMPL_OPENCL
                        , gpuContext
                    #endif
                    );
                    if (!args.filters.empty()) {
                        stft.apply(args.filters);
                    }
                    result = benchmark(
                        warmups,
                        iterations,
                        [&]() { return stft.inverse(); }
                    );
                    break;
                }
                case Phase::FILTER_ONLY: {
                    Stft stft = Stft::forward(
                        input.format(),
                        input.samples()
                    #ifdef IMPL_OPENCL
                        , gpuContext
                    #endif
                    );
                    result = benchmark(
                        warmups,
                        iterations,
                        [&]() {
                            if (!args.filters.empty()) {
                                stft.apply(args.filters);
                            }
                        }
                    );
                    break;
                }
                default:
                    std::unreachable();
            }
            auto ms = [](std::chrono::nanoseconds ns) {
                return static_cast<f64>(ns.count()) / 1'000'000.0;
            };
            std::println(
                "Benchmark ({} warmups, {} iterations, {}):",
                warmups,
                iterations,
                phase
            );
            std::println(
                "┌──────┬─────────────┬─────────────┬─────────────┬────────────"
                    "─┬─────────────┐"
            );
            std::println(
                "│      │ {:^11} │ {:^11} │ {:^11} │ {:^11} │ {:^11} │",
                "Min",
                "Max",
                "Mean",
                "Median",
                "StdDev"
            );
            std::println(
                "├──────┼─────────────┼─────────────┼─────────────┼────────────"
                    "─┼─────────────┤"
            );
            std::println(
                "│ {:<4} │ {:>8.3F} ms │ {:>8.3F} ms │" " {:>8.3F} ms │ "
                    "{:>8.3F} ms │ {:>8.3F} ms │",
                "Wall",
                ms(result.wall.min),
                ms(result.wall.max),
                ms(result.wall.mean),
                ms(result.wall.median),
                ms(result.wall.stdDev)
            );
            std::println(
                "├──────┼─────────────┼─────────────┼─────────────┼────────────"
                    "─┼─────────────┤"
            );
            std::println(
                "│ {:<4} │ {:>8.3F} ms │ {:>8.3F} ms │" " {:>8.3F} ms │ "
                    "{:>8.3F} ms │ {:>8.3F} ms │",
                "CPU",
                ms(result.cpu.min),
                ms(result.cpu.max),
                ms(result.cpu.mean),
                ms(result.cpu.median),
                ms(result.cpu.stdDev)
            );
            std::println(
                "└──────┴─────────────┴─────────────┴─────────────┴────────────"
                    "─┴─────────────┘"
            );
        }
        Stft stft = Stft::forward(
            input.format(),
            input.samples()
        #ifdef IMPL_OPENCL
            , gpuContext
        #endif
        );
        if (!args.filters.empty()) {
            stft.apply(args.filters);
        }
        if (args.output) {
            auto [format, samples] = stft.inverse();
            Wav output(format, std::move(samples));
            output.write(*args.output);
        }
        if (args.spectrum) {
            AudioFormat format = stft.format();
            esap::export_spectrum(
                format.numChannels,
                format.sampleRate,
                stft.num_frames(),
                Stft::WINDOW_SIZE,
                stft.bins(),
                *args.spectrum
            );
        }
        if (args.spectrogram) {
            AudioFormat format = stft.format();
            esap::export_spectrogram(
                format.numChannels,
                format.sampleRate,
                stft.num_frames(),
                Stft::WINDOW_SIZE,
                Stft::HOP_SIZE,
                stft.bins(),
                *args.spectrogram
            );
        }
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::println(stderr, "{}: {}", args.program, e.what());
        return EXIT_FAILURE;
    }
}