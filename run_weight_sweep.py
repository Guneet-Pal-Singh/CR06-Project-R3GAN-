"""Train R3GAN for several R1/R2 settings and calculate FID for each run."""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


WEIGHT_PAIRS = (
    (2.0, 0.5),
    (1.0, 1.0),
    (0.5, 0.5),
    (2.0, 2.0),
)
CSV_FIELDS = (
    "r1_weight",
    "r2_weight",
    "repeat",
    "run_dir",
    "snapshot",
    "fid50k_full",
    "training_time_sec",
    "fid_time_sec",
    "total_time_sec",
)


def run_command(command, log_path):
    """Run a command while displaying and saving its output."""
    print("\n$", " ".join(command), flush=True)
    output = []
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            output.append(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return "".join(output)


def find_run_dir(outdir, output):
    """Read the run directory from train.py output."""
    match = re.search(r"Output directory:\s+(.+)", output)
    if match:
        return Path(match.group(1).strip()).resolve()

    run_dirs = [path for path in outdir.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"Could not find a training run under {outdir}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime).resolve()


def train_one(args, r1_weight, r2_weight, repeat):
    run_started = time.perf_counter()
    command = [
        sys.executable,
        str(args.train_script),
        f"--outdir={args.outdir}",
        f"--data={args.data}",
        f"--r1-weight={r1_weight}",
        f"--r2-weight={r2_weight}",
        f"--gpus={args.gpus}",
        f"--batch={args.batch}",
        f"--mirror={int(args.mirror)}",
        f"--aug={int(args.aug)}",
        f"--cond={int(args.cond)}",
        f"--preset={args.preset}",
        "--metrics=none",
        f"--kimg={args.kimg}",
        f"--tick={args.tick}",
        f"--snap={args.snap}",
    ]
    log_path = args.outdir / f"sweep-r1_{r1_weight:g}-r2_{r2_weight:g}-repeat_{repeat}.log"
    output = run_command(command, log_path)
    run_dir = find_run_dir(args.outdir, output)
    snapshot = run_dir / f"network-snapshot-{args.kimg:09d}.pkl"
    if not snapshot.is_file():
        raise FileNotFoundError(f"Expected snapshot was not created: {snapshot}")
    training_time = time.perf_counter() - run_started

    fid_started = time.perf_counter()
    metric_command = [
        sys.executable,
        str(args.metrics_script),
        "--metrics=fid50k_full",
        f"--network={snapshot}",
        f"--gpus={args.gpus}",
    ]
    metric_output = run_command(metric_command, run_dir / "fid.log")
    fid_time = time.perf_counter() - fid_started
    fid = None
    for line in reversed(metric_output.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if result.get("metric") == "fid50k_full":
            fid = result["results"]["fid50k_full"]
            break
    if fid is None:
        raise ValueError(f"Could not parse fid50k_full from metric output for {snapshot}")

    return {
        "r1_weight": r1_weight,
        "r2_weight": r2_weight,
        "repeat": repeat,
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "fid50k_full": fid,
        "training_time_sec": round(training_time, 3),
        "fid_time_sec": round(fid_time, 3),
        "total_time_sec": round(time.perf_counter() - run_started, 3),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("training-runs"))
    parser.add_argument("--data", type=Path, default=Path("datasets/mnist-32x32.zip"))
    parser.add_argument("--csv", type=Path, default=Path("weight_sweep_results.csv"))
    parser.add_argument("--kimg", type=int, default=500)
    parser.add_argument("--tick", type=int, default=1)
    parser.add_argument("--snap", type=int, default=25)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--preset", default="MNIST-32")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--keep-runs",
        action="store_true",
        help="Keep training run directories after recording their FID",
    )
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cond", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-script", type=Path, default=Path("train.py"))
    parser.add_argument("--metrics-script", type=Path, default=Path("calc_metrics.py"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.data = args.data.resolve()
    args.train_script = args.train_script.resolve()
    args.metrics_script = args.metrics_script.resolve()
    args.csv = args.csv.resolve()

    with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r1_weight, r2_weight in WEIGHT_PAIRS:
            for repeat in range(1, args.repeats + 1):
                result = train_one(args, r1_weight, r2_weight, repeat)
                writer.writerow(result)
                csv_file.flush()
                print(f"Recorded FID {result['fid50k_full']} in {args.csv}", flush=True)
                if not args.keep_runs:
                    run_dir = Path(result["run_dir"])
                    shutil.rmtree(run_dir)
                    print(f"Removed training data: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
