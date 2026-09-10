project(
    'esap',
    'cpp',
    default_options: [
        'cpp_std=c++23',
        'buildtype=debug',
        'b_ndebug=if-release',
        'warning_level=3',
    ],
    meson_version: '>=1.1',
    version: '0.1.0',
)

cpp = meson.get_compiler('cpp')
deps = []

if cpp.get_id() == 'gcc'
    add_project_arguments(
        '-Wundef',
        '-Wshadow=local',
        '-Wconversion',
        '-Wdouble-promotion',
        '-Wcast-align',
        '-Wcast-qual',
        '-Wlogical-op',
        '-Wpointer-arith',
        '-Wformat=2',
        '-Wnrvo',
        '-Wswitch-default',
        '-Wswitch-enum',
        '-Wuseless-cast',
        '-Wmissing-declarations',
        '-Wmissing-include-dirs',
        '-Wduplicated-cond',
        '-Wduplicated-branches',
        '-Wzero-as-null-pointer-constant',
        '-Wunused-macros',
        '-Wenum-conversion',
        '-Wflex-array-member-not-at-end',
        '-Wredundant-decls',
        language: 'cpp',
    )
    if get_option('native')
        add_project_arguments('-march=native', '-mtune=native', language: 'cpp')
    endif
    # Required for some new C++23 features like `std::println()`.
    deps += cpp.find_library('stdc++exp')
endif

incs = include_directories('include')

srcs = files(
    'src/bench.cpp',
    'src/export.cpp',
    'src/main.cpp',
    'src/wav.cpp',
)

impl = get_option('impl')
if impl == 'custom'
    srcs += files('src/custom-stft.cpp')
elif impl == 'fftw'
    srcs += files('src/fftw-stft.cpp')
    deps += dependency('fftw3f')
elif impl == 'opencl'
    add_project_arguments('-DESAP_USE_OPENCL', language: 'cpp')
    srcs += files('src/gpu-context.cpp', 'src/opencl-stft.cpp')
    deps += dependency('OpenCL')
    deps += dependency('VkFFT')
endif

executable('esap', srcs, include_directories: incs, dependencies: deps)