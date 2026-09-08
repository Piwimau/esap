import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

IMPLS = ["custom", "fftw", "opencl", "numpy", "cupy"]

STYLES = {
    "custom": "-",
    "fftw": "-",
    "numpy": "-",
    "opencl": "--",
    "cupy": "--",
}

COLORS = {
    "custom": "#66A5F0",
    "fftw": "#2ED474",
    "numpy": "#52C8EE",
    "opencl": "#E03348",
    "cupy": "#FD7255",
}

MARKERS = {
    "custom": "o",
    "fftw": "s",
    "numpy": "^",
    "opencl": "D",
    "cupy": "v",
}

LABELS = {
    "custom": "Custom",
    "fftw": "FFTW",
    "opencl": "OpenCL",
    "numpy": "NumPy",
    "cupy": "CuPy",
}

REPR_DURATION_S = 60.0

PHASE_LABELS = {
    "forward-only": "Vorwärtstransformation",
    "filter-only": "Nur Filterung",
    "inverse-only": "Inverse Transformation",
    "end-to-end": "Gesamt",
}

PHASE_FILTERS = {
    "forward-only": "none",
    "filter-only": "bandpass",
    "inverse-only": "none",
    "end-to-end": "bandpass",
}


def plot_runtime_scaling(df: pd.DataFrame) -> None:
    """Plots the runtime scaling of the different pipeline implementations
    relative to the audio length.

    Args:
        df: A dataframe containing the benchmark results.
    """
    subset = df[
        (df["phase"] == "end-to-end")
        & (df["sample_rate_hz"] == 48000)
        & (df["channels"] == 1)
        & (df["filter"] == "none")
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for impl in IMPLS:
        impl_df = subset[subset["impl"] == impl].sort_values("duration_s")
        ax.plot(
            impl_df["duration_s"],
            impl_df["wall_median_ms"],
            STYLES[impl],
            color=COLORS[impl],
            marker=MARKERS[impl],
            markersize=5,
            label=LABELS[impl],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Audiolänge (s)")
    ax.set_ylabel("Laufzeit (ms, Median)")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig("runtime-scaling.pdf")


def plot_runtime_speedup(df: pd.DataFrame) -> None:
    """Plots the speedup of the different pipeline implementations relative to
    the custom implementation.

    Args:
        df: A dataframe containing the benchmark results.
    """
    subset = df[
        (df["phase"] == "end-to-end")
        & (df["sample_rate_hz"] == 48000)
        & (df["channels"] == 1)
        & (df["filter"] == "none")
    ]
    baseline = subset[subset["impl"] == "custom"][
        ["duration_s", "wall_median_ms"]
    ].rename(columns={"wall_median_ms": "baseline_ms"})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for impl in IMPLS:
        if impl == "custom":
            continue
        impl_df = subset[subset["impl"] == impl].sort_values("duration_s")
        merged = impl_df.merge(baseline, on="duration_s")
        speedup = merged["baseline_ms"] / merged["wall_median_ms"]
        ax.plot(
            merged["duration_s"],
            speedup,
            STYLES[impl],
            color=COLORS[impl],
            marker=MARKERS[impl],
            markersize=5,
            label=LABELS[impl],
        )
    ax.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Custom",
    )
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Audiolänge (s)")
    ax.set_ylabel("Beschleunigungsfaktor")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig("runtime-speedup.pdf")


def _get_heights(subset: pd.DataFrame, column: str, value) -> np.ndarray:
    heights = []
    for impl in IMPLS:
        row = subset[(subset["impl"] == impl) & (subset[column] == value)]
        heights.append(
            row["wall_median_ms"].iloc[0] if not row.empty else np.nan
        )
    return np.array(heights, dtype=float)


def _plot_ratio_bars(
    ax: plt.Axes,
    subset: pd.DataFrame,
    column: str,
    values: list,
    value_labels: list[str],
    combo_colors: list[str],
) -> None:
    x = np.arange(len(IMPLS))
    width = 0.35
    heights_a = _get_heights(subset, column, values[0])
    heights_b = _get_heights(subset, column, values[1])
    ax.bar(
        x - width / 2,
        heights_a,
        width,
        label=value_labels[0],
        color=combo_colors[0],
    )
    ax.bar(
        x + width / 2,
        heights_b,
        width,
        label=value_labels[1],
        color=combo_colors[1],
    )
    ratios = heights_b / heights_a
    for xi, height_b, ratio in zip(x, heights_b, ratios):
        if np.isnan(ratio):
            continue
        ax.annotate(
            f"x{ratio:.2f}",
            xy=(xi + width / 2, height_b),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.margins(y=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[impl] for impl in IMPLS])
    ax.legend()
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)


def plot_config_comparison(df: pd.DataFrame, duration_s: float) -> None:
    """Plots the runtimes of the different pipeline implementations for
    different sample rates and channel configurations.

    Args:
        df: A dataframe containing the benchmark results.
        duration_s: The representative audio duration (in seconds) for filtering
            the benchmark results.
    """
    combo_colors = ["#B3D2F8", "#66A5F0"]
    fig, (ax_rate, ax_channels) = plt.subplots(
        1,
        2,
        figsize=(10, 4.5),
        sharey=True,
    )
    rate_subset = df[
        (df["phase"] == "end-to-end")
        & (df["filter"] == "none")
        & (df["channels"] == 1)
        & (df["duration_s"] == duration_s)
    ]
    _plot_ratio_bars(
        ax_rate,
        rate_subset,
        "sample_rate_hz",
        [44100, 48000],
        ["44,1 kHz", "48 kHz"],
        combo_colors,
    )
    ax_rate.set_ylabel("Laufzeit (ms, Median)")
    channel_subset = df[
        (df["phase"] == "end-to-end")
        & (df["filter"] == "none")
        & (df["sample_rate_hz"] == 48000)
        & (df["duration_s"] == duration_s)
    ]
    _plot_ratio_bars(
        ax_channels,
        channel_subset,
        "channels",
        [1, 2],
        ["Mono", "Stereo"],
        combo_colors,
    )
    fig.tight_layout()
    fig.savefig("config-comparison.pdf")


def plot_phase_comparison(df: pd.DataFrame, duration_s: float) -> None:
    """Plots the runtimes of the different pipeline implementations for
    different processing phases.

    Args:
        df: A dataframe containing the benchmark results.
        duration_s: The representative audio duration (in seconds) for filtering
            the benchmark results.
    """
    subset = df[
        (df["sample_rate_hz"] == 48000)
        & (df["channels"] == 1)
        & (df["duration_s"] == duration_s)
    ]
    phases = list(PHASE_LABELS)
    x = np.arange(len(phases))
    width = 0.15
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, impl in enumerate(IMPLS):
        heights = []
        for phase in phases:
            row = subset[
                (subset["impl"] == impl)
                & (subset["phase"] == phase)
                & (subset["filter"] == PHASE_FILTERS[phase])
            ]
            heights.append(
                row["wall_median_ms"].iloc[0] if not row.empty else np.nan
            )
        offset = (i - (len(IMPLS) - 1) / 2) * width
        ax.bar(
            x + offset,
            heights,
            width,
            color=COLORS[impl],
            label=LABELS[impl],
        )
    ax.set_yscale("linear")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS[p] for p in phases])
    ax.set_ylabel("Laufzeit (ms, Median)")
    ax.legend()
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig("phase-comparison.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diagrams from the benchmark results."
    )
    parser.add_argument(
        "file",
        help="Path to the CSV file containing the benchmark results.",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.file)
    plot_runtime_scaling(df)
    plot_runtime_speedup(df)
    plot_config_comparison(df, duration_s=REPR_DURATION_S)
    plot_phase_comparison(df, duration_s=REPR_DURATION_S)


if __name__ == "__main__":
    main()
