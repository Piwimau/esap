import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_spectrum(path: str) -> None:
    """Plots a spectrum from a specified CSV file.

    The CSV file is expected to have the columns "channel", "frequency_hz", and
    "magnitude_db". The function will plot the magnitude in dB against the
    frequency in Hz for each unique channel in the data.

    Args:
        path: The path to the CSV file containing the spectrum data.
    """
    df = pd.read_csv(path)
    plt.figure()
    for c in df["channel"].unique():
        c_df = df[df["channel"] == c]
        plt.plot(
            c_df["frequency_hz"], c_df["magnitude_db"], label=f"Channel {c}"
        )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Amplitude Spectrum")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_spectrogram(path: str) -> None:
    """Plots a spectrogram from a specified CSV file.

    The CSV file is expected to have the columns "channel", "time_s", and
    frequency columns in the format "<frequency>hz". The function will plot
    the magnitude in dB against time for each unique channel in the data.

    Args:
        path: The path to the CSV file containing the spectrogram data.
    """
    df = pd.read_csv(path)
    freq_cols = df.columns[2:]
    freqs = [float(c.replace("hz", "")) for c in freq_cols]
    for ch in df["channel"].unique():
        ch_df = df[df["channel"] == ch].sort_values("time_s")
        times = ch_df["time_s"].values
        data = ch_df[freq_cols].values
        plt.figure()
        plt.pcolormesh(times, freqs, data.T, shading="auto", cmap="inferno")
        plt.colorbar(label="Magnitude (dB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Spectrogram (Channel {ch})")
        plt.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a spectrum or spectrogram from a CSV file."
    )
    type_group = parser.add_mutually_exclusive_group(required=True)
    type_group.add_argument(
        "-s",
        "--spectrum",
        dest="type",
        action="store_const",
        const="spectrum",
        help="Plot a spectrum.",
    )
    type_group.add_argument(
        "-S",
        "--spectrogram",
        dest="type",
        action="store_const",
        const="spectrogram",
        help="Plot a spectrogram.",
    )
    parser.add_argument("file", help="Path to the CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Save the plot to this file instead of displaying it.",
    )
    args = parser.parse_args()
    if args.type == "spectrum":
        plot_spectrum(args.file)
    else:
        plot_spectrogram(args.file)
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
