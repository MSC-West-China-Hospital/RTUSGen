import argparse
import os
from PIL import Image
import torch
import torch.nn.functional as F

# 这里假设 physics_guided_losses.py 与本脚本在同一目录
from physics_guided_losses import PhysicsGuidedTextureLoss
from physics_guided_losses import PhysicsGuidedTextureLossNormalized

def load_png_grayscale(path, to_float=True, normalize=True):
    """
    读取单通道PNG为 [1,1,H,W] 的Tensor，范围默认归一化到[0,1]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = Image.open(path).convert('L')  # 灰度
    t = torch.from_numpy((torch.ByteTensor(bytearray(img.tobytes()))
                          .numpy()
                          .reshape(img.size[1], img.size[0]))).float()
    # 或者更直接些：
    # t = torch.from_numpy(np.array(img, dtype=np.float32))
    t = t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    if to_float:
        t = t.float()
    if normalize:
        # 简单min-max到[0,1]
        t_min = t.amin(dim=(2,3), keepdim=True)
        t_max = t.amax(dim=(2,3), keepdim=True)
        t = (t - t_min) / (t_max - t_min + 1e-6)
    return t

def main():
    parser = argparse.ArgumentParser(
        description="Compute Physics-Guided Texture Loss between reference (real US) and predicted (generated US) PNG images."
    )
    parser.add_argument("--ref", required=True, help="Path to reference real ultrasound PNG.")
    parser.add_argument("--pred", required=True, help="Path to predicted/generated ultrasound PNG.")
    parser.add_argument("--bmode", action="store_true",
                        help="Set if your input PNGs are B-mode (log compressed).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cuda or cpu")
    # 一些可调超参（保持与模块默认一致即可）
    parser.add_argument("--w_radial_psd", type=float, default=1.0)
    parser.add_argument("--w_slope", type=float, default=0.2)
    parser.add_argument("--w_intercept", type=float, default=0.2)
    parser.add_argument("--w_auto_axial", type=float, default=0.5)
    parser.add_argument("--w_auto_lat", type=float, default=0.5)
    parser.add_argument("--w_nkg_m", type=float, default=0.5)
    parser.add_argument("--w_nkg_omega", type=float, default=0.25)
    parser.add_argument("--w_alpha", type=float, default=0.2)
    parser.add_argument("--w_gram", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device(args.device)

    # 读取图像
    I_ref = load_png_grayscale(args.ref).to(device)   # [1,1,H,W]
    I_pred = load_png_grayscale(args.pred).to(device) # [1,1,H,W]

    # 尺寸不一致时，做一个安全的 resize 到 ref 尺寸
    if I_ref.shape != I_pred.shape:
        I_pred = F.interpolate(I_pred, size=I_ref.shape[-2:], mode="bilinear", align_corners=False)

    # # 初始化 Physics-Guided 损失器
    # phy_loss = PhysicsGuidedTextureLoss(
    #     w_radial_psd=args.w_radial_psd,
    #     w_slope=args.w_slope, w_intercept=args.w_intercept,
    #     w_auto_axial=args.w_auto_axial, w_auto_lat=args.w_auto_lat,
    #     w_nkg_m=args.w_nkg_m, w_nkg_omega=args.w_nkg_omega,
    #     w_alpha=args.w_alpha,
    #     w_gram=args.w_gram,
    #     use_window=True,
    #     slope_range=(0.1, 0.5),
    #     gram_scales=(7, 11, 15),
    #     gram_thetas=(0, 0.52, 1.05, 1.57, 2.09, 2.62),  # 约等于 0,30,60,90,120,150 度
    #     gram_sigma=2.0,
    #     nkg_win=32, nkg_stride=16
    # ).to(device)

    # # 计算损失（若是B-mode图像，传 inputs_are_bmode=True）
    # # 计算损失
    # with torch.no_grad():
    #     loss_val, loss_dict = phy_loss(I_ref, I_pred, inputs_are_bmode=args.bmode, return_dict=True)
    
    # print("=== Physics-Guided Texture Loss ===")
    # print(f"Ref:  {args.ref}")
    # print(f"Pred: {args.pred}")
    # print(f"Inputs are B-mode: {args.bmode}")
    # print(f"Total loss: {loss_val.item():.6f}")
    # print("--- Sub losses ---")
    # for k, v in loss_dict.items():
    #     print(f"{k:25s}: {v:.6f}")

    # ...
    phy_loss = PhysicsGuidedTextureLossNormalized(
        w_radial_psd=1.0,
        w_slope=0.2, w_intercept=0.2,
        w_auto_axial=0.5, w_auto_lat=0.5,
        w_nkg_m=0.5, w_nkg_omega=0.25,
        w_alpha=0.2,
        w_gram=0.2,
        use_window=True,
        slope_range=(0.1, 0.5),
        gram_scales=(7,11,15),
        gram_thetas=(0, 0.52, 1.05, 1.57, 2.09, 2.62),
        gram_sigma=2.0,
        nkg_win=32, nkg_stride=16,
        att_rel_eps=1e-3
    ).to(device)
    
    with torch.no_grad():
        total, d = phy_loss(I_ref, I_pred, inputs_are_bmode=args.bmode, return_dict=True)
    
    print("=== Physics-Guided Texture Loss (Normalized) ===")
    print(f"Total: {total.item():.6f}")
    for k, v in d.items():
        print(f"{k:24s}: {v:.6f}")

if __name__ == "__main__":
    main()
