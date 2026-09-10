import argparse
import csv
import itertools
import json
import numpy as np
import os
import shutil
import subprocess
import wave

from dataclasses import dataclass


@dataclass
class Round:
    duration_s: float
    warmups: int
    iterations: int


IMPLS = ["custom", "fftw", "opencl"]

PHASES = [
    ("end-to-end", False),
    ("end-to-end", True),
    ("forward-only", False),
    ("inverse-only", False),
    ("filter-only", True),
]

ROUNDS = [
    Round(duration_s=0.1, warmups=15, iterations=300),
    Round(duration_s=0.25, warmups=15, iterations=250),
    Round(duration_s=0.5, warmups=10, iterations=200),
    Round(duration_s=1, warmups=10, iterations=150),
    Round(duration_s=5, warmups=10, iterations=100),
    Round(duration_s=10, warmups=10, iterations=75),
    Round(duration_s=30, warmups=5, iterations=50),
    Round(duration_s=60, warmups=5, iterations=30),
    Round(duration_s=120, warmups=3, iterations=20),
    Round(duration_s=180, warmups=3, iterations=15),
    Round(duration_s=240, warmups=2, iterations=10),
    Round(duration_s=300, warmups=2, iterations=10),
]

SAMPLE_RATES = [44100, 48000]

CHANNELS = [1, 2]

CONFIGS = itertools.product(IMPLS, PHASES, ROUNDS, SAMPLE_RATES, CHANNELS)

INT16_MIN = -32768
INT16_MAX = 32767

FIELD_NAMES = [
    "impl",
    "phase",
    "duration_s",
    "sample_rate_hz",
    "channels",
    "filter",
    "wall_min_ms",
    "wall_max_ms",
    "wall_mean_ms",
    "wall_median_ms",
    "wall_stddev_ms",
    "cpu_min_ms",
    "cpu_max_ms",
    "cpu_mean_ms",
    "cpu_median_ms",
    "cpu_stddev_ms",
]


def generate_audio_files(audio_dir: str) -> None:
    """Generates white noise audio files with various durations, sample rates,
    and channel configurations.

    The generated audio files are saved in the specified directory. Each file is
    named according to its duration, sample rate, and channel configuration
    (e.g., "wn-1s-44100Hz-mono.wav").

    Args:
        audio_dir: The directory where the generated audio files will be saved.
    """
    print(f"Generating audio files in '{audio_dir}'...")
    os.makedirs(audio_dir, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    num_files = len(ROUNDS) * len(SAMPLE_RATES) * len(CHANNELS)
    idx = 1
    for round in ROUNDS:
        for sample_rate in SAMPLE_RATES:
            for channels in CHANNELS:
                num_samples = int(round.duration_s * sample_rate)
                normalized_samples = rng.uniform(
                    low=-1.0, high=1.0, size=(num_samples, channels)
                )
                pcm_samples = (
                    (normalized_samples * INT16_MAX)
                    .clip(INT16_MIN, INT16_MAX)
                    .astype(np.int16)
                )
                frames = pcm_samples.tobytes()
                channel_desc = "mono" if channels == 1 else "stereo"
                name = (
                    f"wn-{round.duration_s}s-{sample_rate}Hz-{channel_desc}.wav"
                )
                path = os.path.join(audio_dir, name)
                print(f"({idx}/{num_files}) Generating '{path}'...")
                with wave.open(path, "w") as f:
                    f.setnchannels(channels)
                    f.setsampwidth(2)
                    f.setframerate(sample_rate)
                    f.writeframes(frames)
                idx += 1
    print()


def build_binaries(binaries_dir: str) -> dict[str, str]:
    """Builds the binaries for the different implementations of the audio
    processing pipeline.

    The built binaries are saved in the specified directory. Each binary is
    named according to its implementation (e.g., "esap-custom", "esap-fftw",
    "esap-opencl").

    Args:
        binaries_dir: The directory where the built binaries will be saved.

    Returns:
        A dictionary mapping each implementation to its binary path.
    """
    print(f"Building binaries in '{binaries_dir}'...")
    os.makedirs(binaries_dir, exist_ok=True)
    binaries = {}
    for i, impl in enumerate(IMPLS):
        name = f"esap-{impl}"
        dst = os.path.join(binaries_dir, name)
        print(f"({i + 1}/{len(IMPLS)}) Building '{dst}'...")
        subprocess.run(
            [
                "meson",
                "setup",
                "build",
                "--buildtype=release",
                "-Dnative=true",
                f"-Dimpl={impl}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            ["meson", "compile", "-C", "build"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        src = os.path.join("build", "esap")
        shutil.copy2(src, dst)
        subprocess.run(
            ["rm", "-rf", "build"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        binaries[impl] = dst
    print()
    return binaries


def run_benchmarks(
    audio_dir: str,
    binaries: dict[str, str],
    results_dir: str,
) -> None:
    """Runs the benchmarks for each implementation of the audio processing
    pipeline using the generated audio files in a specified directory.

    Args:
        audio_dir: The directory containing the generated audio files.
        binaries: A dictionary mapping each implementation to its binary path.
        results_dir: The directory where the benchmark results will be saved.
    """
    print("Running benchmarks...")
    os.makedirs(results_dir, exist_ok=True)
    num_benchmarks = (
        len(IMPLS)
        * len(PHASES)
        * len(ROUNDS)
        * len(SAMPLE_RATES)
        * len(CHANNELS)
    )
    idx = 1
    results_path = os.path.join(results_dir, "results.csv")
    with open(results_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()
        for (
            impl,
            (phase, use_filter),
            round,
            sample_rate,
            channels,
        ) in CONFIGS:
            channel_desc = "mono" if channels == 1 else "stereo"
            audio_name = (
                f"wn-{round.duration_s}s-{sample_rate}Hz-{channel_desc}.wav"
            )
            audio_path = os.path.join(audio_dir, audio_name)
            filter_desc = "bandpass" if use_filter else "none"
            print(
                f"({idx}/{num_benchmarks}) Benchmarking '{audio_path}' "
                + f"(impl={impl}, phase={phase}, filter={filter_desc})..."
            )
            cmd = [
                binaries[impl],
                "--benchmark",
                "--bench-format=json",
                f"--warmups={round.warmups}",
                f"--iterations={round.iterations}",
                f"--phase={phase}",
            ]
            if use_filter:
                cmd.append("--filter=bandpass:300:3400")
            cmd.append(audio_path)
            result = subprocess.run(cmd, capture_output=True, text=True)
            idx += 1
            if result.returncode != 0:
                print(
                    f"Error: Benchmark with '{audio_path}' (impl={impl}, "
                    f"phase={phase}, filter={filter_desc}) failed."
                )
                print("Standard Output:")
                print(result.stdout)
                print("Standard Error:")
                print(result.stderr)
                continue
            data = json.loads(result.stdout)
            writer.writerow(
                {
                    "impl": impl,
                    "phase": phase,
                    "duration_s": round.duration_s,
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                    "filter": filter_desc,
                    "wall_min_ms": data["wall"]["min_ms"],
                    "wall_max_ms": data["wall"]["max_ms"],
                    "wall_mean_ms": data["wall"]["mean_ms"],
                    "wall_median_ms": data["wall"]["median_ms"],
                    "wall_stddev_ms": data["wall"]["stdDev_ms"],
                    "cpu_min_ms": data["cpu"]["min_ms"],
                    "cpu_max_ms": data["cpu"]["max_ms"],
                    "cpu_mean_ms": data["cpu"]["mean_ms"],
                    "cpu_median_ms": data["cpu"]["median_ms"],
                    "cpu_stddev_ms": data["cpu"]["stdDev_ms"],
                }
            )
            f.flush()
    print(f"Benchmark results saved to '{results_path}'.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the performance of the audio processing "
        "pipeline by measuring the execution time of processing a set of "
        "generated audio files."
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="benchmark",
        help="Specify the directory for saving the generated audio files and "
        "benchmark results (default: './benchmark').",
    )
    args = parser.parse_args()
    audio_dir = os.path.join(args.dir, "resources")
    generate_audio_files(audio_dir)
    binaries_dir = os.path.join(args.dir, "bin")
    binaries = build_binaries(binaries_dir)
    run_benchmarks(audio_dir, binaries, args.dir)


if __name__ == "__main__":
    main()
