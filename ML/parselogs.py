#!/usr/bin/env python3

import sys
import shlex
import re
import csv

def parse_first_line_for_params(line):
    kernel1 = None
    kernel2 = None
    latentdim = None
    epochs = None
    dropout = None
    pooling = None
    activation = None
    unet_model = None
    loss = None

    if "Commandline:" not in line:
        return kernel1, kernel2, latentdim, epochs, dropout, pooling, activation, unet_model, loss

    cmd_part = line.split("Commandline:", 1)[1].strip()

    try:
        args = shlex.split(cmd_part)
    except ValueError:
        args = cmd_part.split()

    def get_arg_value(flag):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                return args[idx + 1]
        return None

    kernel1 = get_arg_value("--kernel1")
    kernel2 = get_arg_value("--kernel2")
    latentdim = get_arg_value("--latentdim")
    epochs   = get_arg_value("--epochs")
    dropout  = get_arg_value("--dropout")
    pooling  = get_arg_value("--pooling")
    activation  = get_arg_value("--activation")
    unet_model  = get_arg_value("--unet_model")
    loss  = get_arg_value("--loss")

    return kernel1, kernel2, latentdim, epochs, dropout, pooling, activation, unet_model, loss


def extract_max_r2_from_file(lines):
    r2_pattern = re.compile(r"R2 Score:\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)")
    max_r2 = None

    for line in lines:
        m = r2_pattern.search(line)
        if m:
            try:
                v = float(m.group(1))
                if max_r2 is None or v > max_r2:
                    max_r2 = v
            except:
                continue
    return max_r2


def process_log_file(filename):
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
    except:
        return None

    if not lines:
        return None

    kernel1, kernel2, latentdim, epochs, dropout, pooling, activation, unet_model, loss = parse_first_line_for_params(lines[0])

    # defaults
    if kernel1 is None:   kernel1   = "5"
    if kernel2 is None:   kernel2   = "3"
    if latentdim is None: latentdim = "256"
    if dropout is None:   dropout   = "0.2"
    if epochs is None:    epochs    = ""
    if pooling is None:    pooling    = "max"
    if activation is None:    activation    = "relu"
    if unet_model is None:    unet_model    = "skip"
    if loss is None:    loss    = "chisq"

    max_r2 = extract_max_r2_from_file(lines)
    if max_r2 is None:
        return None

    return {
        "filename": filename,
        "kernel1": kernel1,
        "kernel2": kernel2,
        "latentdim": latentdim,
        "epochs": epochs,
        "dropout": dropout,
        "pooling": pooling,
        "activation": activation,
        "unet_model": unet_model,
        "loss": loss,
        "Max_R2": max_r2
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} log1 log2 ...", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fn in sys.argv[1:]:
        r = process_log_file(fn)
        if r is not None:
            rows.append(r)

    # SORT ASCENDING
    rows.sort(key=lambda x: x["Max_R2"])

    writer = csv.writer(sys.stdout)
    writer.writerow(["filename", "kernel1", "kernel2", "latentdim", "epochs", "dropout", "pooling", "activation", "unet_model", "loss", "Max_R2"])

    for r in rows:
        writer.writerow([
            r["filename"],
            r["kernel1"],
            r["kernel2"],
            r["latentdim"],
            r["epochs"],
            r["dropout"],
            r["pooling"],
            r["activation"],
            r["unet_model"],
            r["loss"],
            f"{r['Max_R2']:.6f}"
        ])


if __name__ == "__main__":
    main()
