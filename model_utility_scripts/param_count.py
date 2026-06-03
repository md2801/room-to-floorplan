import csv
import torch
import matplotlib.pyplot as plt

from room_to_floorplan.roomtofp_polygon_model import RoomToFPPolygonNet

CHECKPOINT = r"C:\Users\aaqil\Work\Masters\Academics\room_to_floorplan_project\output\v7\artifacts\roomtofp_polygon_v7_512_c16_final\best_ema.pt"

def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["ema", "model", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt

def clean_state_dict_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned

model = RoomToFPPolygonNet(
    in_channels=8,
    max_rooms=20,
    max_corners=16,
    embed_dim=256,
    room_classes=12,
    dropout=0.10,
    predict_furniture=False,
    base=48,
    mask_dim=128,
    decoder_layers=4,
    corner_decoder_layers=3,
    num_heads=8,
    transformer_encoder_layers=2,
    deform_points=4,
)

ckpt = torch.load(CHECKPOINT, map_location="cpu")
state_dict = clean_state_dict_keys(get_state_dict(ckpt))

missing, unexpected = model.load_state_dict(state_dict, strict=False)

print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

rows = []

for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

    if params > 0:
        rows.append({
            "module": name,
            "parameters": params,
            "trainable_parameters": trainable,
            "percentage": params / total_params * 100,
        })

rows = sorted(rows, key=lambda r: r["parameters"], reverse=True)

print("\nParameter breakdown:")
print("-" * 80)
for r in rows:
    print(f"{r['module']:30s} {r['parameters']:15,} {r['percentage']:8.2f}%")

with open("roomtofp_v7_parameter_breakdown.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["module", "parameters", "trainable_parameters", "percentage"]
    )
    writer.writeheader()
    writer.writerows(rows)

module_names = [r["module"] for r in rows]
param_counts = [r["parameters"] for r in rows]

plt.figure(figsize=(12, 7))
plt.barh(module_names, param_counts)
plt.xlabel("Number of parameters")
plt.ylabel("Module")
plt.title("RoomToFP V7 parameter count by module")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("roomtofp_v7_parameter_breakdown.png", dpi=200)
plt.show()

print("\nSaved:")
print("roomtofp_v7_parameter_breakdown.csv")
print("roomtofp_v7_parameter_breakdown.png")