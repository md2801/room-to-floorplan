from pathlib import Path
import numpy as np

root = Path("output/data/bev_pairs")

deleted = 0
checked = 0

for obs_file in root.glob("*_obs.npy"):
    target_file = Path(str(obs_file).replace("_obs.npy", "_target.npy"))

    try:
        x = np.load(obs_file)
        y = np.load(target_file)

        # sanity check shapes
        if x.shape != (512, 512, 6) or y.shape != (2, 512, 512):
            raise ValueError("wrong shape")

    except Exception:
        print(f"[DELETE] {obs_file.name}")
        obs_file.unlink(missing_ok=True)
        target_file.unlink(missing_ok=True)
        deleted += 1

    checked += 1

print("=" * 50)
print(f"Checked: {checked}")
print(f"Deleted: {deleted}")