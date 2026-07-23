import torch
import yaml
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/teacher_large_1/best_model.pt"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("=== Thông tin checkpoint ===")
print(f"Epoch: {ckpt.get('epoch', '?')}")
print(f"Val F1 lúc train: {ckpt.get('val_f1', '?')}")
print(f"Val metrics: {ckpt.get('val_metrics', '?')}")
print()

if "config" in ckpt:
    print("=== CONFIG ĐẦY ĐỦ ĐÃ DÙNG ĐỂ TRAIN CHECKPOINT NÀY ===")
    print(yaml.dump(ckpt["config"], allow_unicode=True, default_flow_style=False, sort_keys=False))

    # Lưu ra file để dùng lại / tái lập
    out_path = "recovered_supcon_teacher_config.yaml"
    with open(out_path, "w") as f:
        yaml.dump(ckpt["config"], f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"\n✓ Đã lưu ra: {out_path}")
else:
    print("✗ Checkpoint này KHÔNG có key 'config' — có thể được lưu bởi phiên bản train_teacher.py cũ hơn.")