from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def validate_sample(obs_file: Path, target_file: Path) -> dict:
    result = {
        "obs_file": obs_file.name,
        "target_file": target_file.name,
        "ok": True,
        "issues": [],
        "obs_shape": None,
        "target_shape": None,
        "obs_min": None,
        "obs_max": None,
        "target_unique": None,
        "structure_pixels": None,
        "furniture_pixels": None,
    }

    try:
        obs = np.load(obs_file)
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"failed_to_load_obs: {type(e).__name__}: {e}")
        return result

    try:
        target = np.load(target_file)
    except Exception as e:
        result["ok"] = False
        result["issues"].append(f"failed_to_load_target: {type(e).__name__}: {e}")
        return result

    result["obs_shape"] = tuple(obs.shape)
    result["target_shape"] = tuple(target.shape)

    if obs.ndim != 3:
        result["ok"] = False
        result["issues"].append(f"obs_ndim_expected_3_got_{obs.ndim}")

    if target.ndim != 3:
        result["ok"] = False
        result["issues"].append(f"target_ndim_expected_3_got_{target.ndim}")

    if obs.ndim == 3 and obs.shape[-1] != 6:
        result["issues"].append(f"obs_last_dim_expected_6_got_{obs.shape[-1]}")

    if target.ndim == 3 and target.shape[0] != 2:
        result["issues"].append(f"target_first_dim_expected_2_got_{target.shape[0]}")

    if np.isnan(obs).any() or np.isinf(obs).any():
        result["ok"] = False
        result["issues"].append("obs_contains_nan_or_inf")

    if np.isnan(target).any() or np.isinf(target).any():
        result["ok"] = False
        result["issues"].append("target_contains_nan_or_inf")

    obs_min = float(obs.min())
    obs_max = float(obs.max())
    result["obs_min"] = obs_min
    result["obs_max"] = obs_max

    if obs_min < -1e-6 or obs_max > 1.0 + 1e-6:
        result["issues"].append(f"obs_out_of_expected_range_{obs_min:.6f}_{obs_max:.6f}")

    uniq = np.unique(target)
    result["target_unique"] = uniq.tolist()[:20]

    if not np.all(np.isin(uniq, [0.0, 1.0])):
        result["issues"].append("target_not_binary")

    if target.ndim == 3 and target.shape[0] >= 2:
        result["structure_pixels"] = float(target[0].sum())
        result["furniture_pixels"] = float(target[1].sum())

        if result["structure_pixels"] == 0:
            result["issues"].append("empty_structure_target")
        if result["furniture_pixels"] == 0:
            result["issues"].append("empty_furniture_target")

    return result


def visualize_samples(root: Path, obs_files: list[Path], max_visualize: int) -> None:
    if not obs_files or max_visualize <= 0:
        return

    chosen = random.sample(obs_files, min(max_visualize, len(obs_files)))

    for obs_file in chosen:
        target_file = Path(str(obs_file).replace("_obs.npy", "_target.npy"))
        obs = np.load(obs_file)
        target = np.load(target_file)

        occ = obs[..., 0]
        structure_in = obs[..., 1]
        furniture_in = obs[..., 2]
        rgb = obs[..., 3:6]

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(obs_file.stem)

        ax = plt.subplot(2, 3, 1)
        ax.imshow(occ, cmap="gray")
        ax.set_title("Occupancy")
        ax.axis("off")

        ax = plt.subplot(2, 3, 2)
        ax.imshow(structure_in, cmap="gray")
        ax.set_title("Structure input")
        ax.axis("off")

        ax = plt.subplot(2, 3, 3)
        ax.imshow(furniture_in, cmap="gray")
        ax.set_title("Furniture input")
        ax.axis("off")

        ax = plt.subplot(2, 3, 4)
        ax.imshow(rgb)
        ax.set_title("RGB BEV")
        ax.axis("off")

        ax = plt.subplot(2, 3, 5)
        ax.imshow(target[0], cmap="gray")
        ax.set_title("Target structure")
        ax.axis("off")

        ax = plt.subplot(2, 3, 6)
        ax.imshow(target[1], cmap="gray")
        ax.set_title("Target furniture")
        ax.axis("off")

        plt.tight_layout()

        viz_dir = root / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        out_path = viz_dir / f"{obs_file.stem}_viz.png"
        plt.savefig(out_path)
        plt.close()

        print(f"[VIS] Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--visualize", type=int, default=0,
                    help="Number of random samples to visualize")
    args = ap.parse_args()

    root = Path(args.data_dir)
    obs_files = sorted(root.glob("*_obs.npy"))

    print("=" * 80)
    print("VALIDATE BEV DATASET")
    print("=" * 80)
    print(f"Data dir: {root.resolve()}")
    print(f"Observation files found: {len(obs_files)}")
    print("=" * 80)

    if not obs_files:
        print("[FATAL] No *_obs.npy files found.")
        return

    missing_targets = []
    results = []
    obs_shapes = {}
    target_shapes = {}
    bad_samples = []
    empty_structure = 0
    empty_furniture = 0

    for obs_file in obs_files:
        target_file = Path(str(obs_file).replace("_obs.npy", "_target.npy"))
        if not target_file.exists():
            missing_targets.append(obs_file.name)
            continue

        result = validate_sample(obs_file, target_file)
        results.append(result)

        if result["obs_shape"] is not None:
            obs_shapes[str(result["obs_shape"])] = obs_shapes.get(str(result["obs_shape"]), 0) + 1
        if result["target_shape"] is not None:
            target_shapes[str(result["target_shape"])] = target_shapes.get(str(result["target_shape"]), 0) + 1

        if "empty_structure_target" in result["issues"]:
            empty_structure += 1
        if "empty_furniture_target" in result["issues"]:
            empty_furniture += 1

        if not result["ok"] or result["issues"]:
            bad_samples.append(result)

    total_paired = len(results)

    summary = {
        "data_dir": str(root.resolve()),
        "obs_files_found": len(obs_files),
        "paired_samples": total_paired,
        "missing_targets": missing_targets,
        "missing_targets_count": len(missing_targets),
        "obs_shapes": obs_shapes,
        "target_shapes": target_shapes,
        "empty_structure_count": empty_structure,
        "empty_furniture_count": empty_furniture,
        "empty_structure_ratio": empty_structure / max(total_paired, 1),
        "empty_furniture_ratio": empty_furniture / max(total_paired, 1),
        "samples_with_issues_count": len(bad_samples),
        "samples_with_issues": bad_samples[:200],
    }

    print("[SUMMARY]")
    print(f"Paired samples: {summary['paired_samples']}")
    print(f"Missing targets: {summary['missing_targets_count']}")
    print(f"Observation shapes: {summary['obs_shapes']}")
    print(f"Target shapes: {summary['target_shapes']}")
    print(f"Empty structure targets: {summary['empty_structure_count']} "
          f"({summary['empty_structure_ratio']:.3f})")
    print(f"Empty furniture targets: {summary['empty_furniture_count']} "
          f"({summary['empty_furniture_ratio']:.3f})")
    print(f"Samples with issues: {summary['samples_with_issues_count']}")
    print("=" * 80)

    if summary["missing_targets_count"] > 0:
        print("[MISSING TARGETS] First few:")
        for name in missing_targets[:20]:
            print(f"  - {name}")
        print("=" * 80)

    if bad_samples:
        print("[SAMPLES WITH ISSUES] First few:")
        for item in bad_samples[:20]:
            print(f"- {item['obs_file']}")
            for issue in item["issues"]:
                print(f"    * {issue}")
        print("=" * 80)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[REPORT] Saved JSON report to: {report_path}")

    if args.visualize > 0:
        visualize_samples(root, obs_files, args.visualize)


if __name__ == "__main__":
    main()