#!python
import re
import csv
from pathlib import Path

# Regex to extract fX and xHI from filenames
pattern = re.compile(r"fX(-?\d+\.\d+)_xHI(\d+\.\d+)")

def parse_file(input_file, output_csv):
    rows = []

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract filename part
            # (everything after the permissions, so split by spaces and take the last token)
            filename = line.split()[-1]

            m = pattern.search(filename)
            if m:
                fX = float(m.group(1))
                xHI = float(m.group(2))
                rows.append([filename, fX, xHI])
            else:
                print(f"Warning: Could not extract fX/xHI from: {filename}")

    # Sort by fX first, then xHI
    rows.sort(key=lambda x: (x[1], x[2]))

    # Write CSV
    with open(output_csv, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["filename", "fX", "xHI"])
        writer.writerows(rows)

    print(f"Written: {output_csv} ({len(rows)} rows)")

import pandas as pd
import matplotlib.pyplot as plt

def plot_xhi_fx(training_csv="training_parsed.csv", test_csv="test_parsed.csv"):
    # Read CSV files
    df_train = pd.read_csv(training_csv)
    df_test = pd.read_csv(test_csv)

    # Extract values
    x_train, y_train = df_train["xHI"], df_train["fX"]
    x_test, y_test = df_test["xHI"], df_test["fX"]

    # Plot
    plt.figure(figsize=(8, 6))

    # Training points: circles
    plt.scatter(x_train, y_train, marker='o', label="Training", alpha=0.7)

    # Test points: stars
    plt.scatter(x_test, y_test, marker='*', s=120, label="Test", alpha=0.9)

    # Axis limits
    plt.xlim(0, 1)
    plt.ylim(-4, 1)

    plt.xlabel("xHI", fontsize=12)
    plt.ylabel("fX", fontsize=12)
    plt.title("Scatter Plot of xHI vs fX", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("tmp_out/training_test_points.png", format="png", dpi=300)

# Process the two files
parse_file("tmp_out/training_files.txt", "tmp_out/training_parsed.csv")
parse_file("tmp_out/test_files.txt", "tmp_out/test_parsed.csv")

plot_xhi_fx("tmp_out/training_parsed.csv", "tmp_out/test_parsed.csv")

