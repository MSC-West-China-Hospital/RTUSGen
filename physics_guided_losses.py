import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 工具函数
# ---------------------------

def _hann2d(H: int, W: int, device):
    win_h = torch.hann_window(H, device=device).unsqueeze(1)
    win_w = torch.hann_window(W, device=device).unsqueeze(0)
    return (win_h @ win_w)  # [H,W]


def _fft_power_spectrum(img: torch.Tensor, eps=1e-6, window=True):
    """
    img: [B,1,H,W]  -- 建议为包络或B-mode; 使用前可减均值
    返回：log功率谱 [B,1,H,W]
    """
    B, C, H, W = img.shape
    x = img - img.mean(dim=(2, 3), keepdim=True)
    if window:
        win = _hann2d(H, W, img.device).unsqueeze(0).unsqueeze(0)
        x = x * win
    F2 = torch.fft.fft2(x, norm='ortho')
    P = (F2.real ** 2 + F2.imag ** 2).clamp_min(eps)
    return torch.log(P + eps)


def _radial_average(ps_log: torch.Tensor):
    """
    ps_log: [B,1,H,W]  (对数功率谱)
    返回：径向平均后的谱 [B, R]  (R ~ 半径bin数)
    """
    B, _, H, W = ps_log.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=ps_log.device),
        torch.arange(W, device=ps_log.device),
        indexing='ij'
    )
    cy, cx = (H - 1) / 2, (W - 1) / 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).round().long()
    r_max = int(r.max().item())
    r_flat = r.view(-1)

    radials = []
    one = torch.ones((H * W,), device=ps_log.device)
    for b in range(B):
        arr = ps_log[b, 0].contiguous().view(-1)
        sum_r = torch.zeros(r_max + 1, device=ps_log.device)
        cnt_r = torch.zeros(r_max + 1, device=ps_log.device)
        sum_r = sum_r.scatter_add(0, r_flat, arr)
        cnt_r = cnt_r.scatter_add(0, r_flat, one)
        rad = sum_r / (cnt_r + 1e-6)
        radials.append(rad.unsqueeze(0))
    return torch.cat(radials, dim=0)  # [B,R]


def _standardize(x: torch.Tensor, dim=-1, eps=1e-6):
    m = x.mean(dim=dim, keepdim=True)
    s = x.std(dim=dim, keepdim=True)
    return (x - m) / (s + eps)


def _fit_line_ls(x: torch.Tensor, y: torch.Tensor):
    """
    对每个 batch 做简单的线性回归 y ~ a*x + b
    x,y: [B,N]
    返回 a,b: [B]
    """
    B, N = x.shape
    x1 = torch.stack([x, torch.ones_like(x)], dim=2)  # [B,N,2]
    # 正规方程：(X^T X)^{-1} X^T y
    xtx = torch.matmul(x1.transpose(1, 2), x1)        # [B,2,2]
    xty = torch.matmul(x1.transpose(1, 2), y.unsqueeze(2))  # [B,2,1]
    inv = torch.linalg.inv(xtx + 1e-6 * torch.eye(2, device=x.device).unsqueeze(0))
    ab = torch.matmul(inv, xty).squeeze(2)  # [B,2]
    a, b = ab[:, 0], ab[:, 1]
    return a, b


def _autocorr2d(img: torch.Tensor, eps=1e-6):
    """
    归一化自相关：IFFT(|FFT|^2)，再归一化到[0,1]
    img: [B,1,H,W]
    返回：R: [B,1,H,W]
    """
    B, _, H, W = img.shape
    x = img - img.mean(dim=(2, 3), keepdim=True)
    F2 = torch.fft.fft2(x, norm='ortho')
    P = (F2.real ** 2 + F2.imag ** 2)
    R = torch.fft.ifft2(P, norm='ortho').real
    R = R / (R.amax(dim=(2, 3), keepdim=True).clamp_min(eps))
    return R


def _half_power_width(curve: torch.Tensor, axis=-1):
    """
    估计半功率宽度（-3dB宽度）。curve ~ 以中心峰为1，向两侧衰减。
    输入 curve: [B,L]，返回宽度 [B]
    用 soft-argmin/线性插值做可微近似（这里用简单线性插值+裁剪）
    """
    B, L = curve.shape
    peak = curve.amax(dim=1, keepdim=True).clamp_min(1e-6)
    target = 0.5 * peak  # 半功率
    # 左侧
    idx = torch.arange(L, device=curve.device).float().unsqueeze(0).expand(B, -1)
    # 找到最接近 target 的位置（简化近似）
    diff = (curve - target).abs()
    argmin = diff.argmin(dim=1).float()  # [B]  (非严格可微，但稳定)
    # 宽度近似为 2*argmin（假设中心在0）
    # 更严格可将中心对齐后再测，这里简化处理
    return 2.0 * argmin


def _gabor_kernels(scales: List[int], thetas: List[float], sigma: float = 2.0):
    """
    生成一组Gabor核（实部cos分量），将不同k的核零填充到相同的max_k×max_k，再cat。
    返回: [N, 1, max_k, max_k]
    """
    assert all(s % 2 == 1 for s in scales), "All scales must be odd."
    max_k = max(scales)
    kernels = []

    for k in scales:
        half = k // 2
        yy, xx = torch.meshgrid(
            torch.arange(-half, half + 1),
            torch.arange(-half, half + 1),
            indexing='ij'
        )
        yy = yy.float()
        xx = xx.float()
        for th in thetas:
            x_theta = xx * math.cos(th) + yy * math.sin(th)
            y_theta = -xx * math.sin(th) + yy * math.cos(th)
            gb = torch.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2)) * torch.cos(2 * math.pi * x_theta / k)
            gb = gb - gb.mean()
            gb = gb / (gb.norm() + 1e-6)

            pad_total = max_k - k
            pad_each  = pad_total // 2
            gb_pad = F.pad(gb, (pad_each, pad_each, pad_each, pad_each))
            if gb_pad.shape[0] != max_k or gb_pad.shape[1] != max_k:
                gb_pad = F.pad(gb_pad, (0, max_k - gb_pad.shape[1], 0, max_k - gb_pad.shape[0]))
            kernels.append(gb_pad.unsqueeze(0).unsqueeze(0))  # [1,1,max_k,max_k]

    return torch.cat(kernels, dim=0)  # [N,1,max_k,max_k]




def _gram_matrix(feat: torch.Tensor):
    """
    feat: [B,N,H,W]
    返回 Gram: [B,N,N]
    """
    B, N, H, W = feat.shape
    Ff = feat.view(B, N, H * W)
    G = torch.matmul(Ff, Ff.transpose(1, 2)) / (H * W)
    return G


# ---------------------------
# 1) 频谱一致性（径向功率谱）
# ---------------------------

def radial_spectrum_loss(I_ref: torch.Tensor, I_pred: torch.Tensor, window=True, w_mse=1.0):
    """
    让 speckle 的空间频谱一致（避免假纹理）
    返回：标量loss
    """
    ps_ref = _fft_power_spectrum(I_ref, window=window)  # [B,1,H,W]
    ps_pred = _fft_power_spectrum(I_pred, window=window)
    r_ref = _radial_average(ps_ref)  # [B,R]
    r_pred = _radial_average(ps_pred)
    r_ref = _standardize(r_ref, dim=1)
    r_pred = _standardize(r_pred, dim=1)
    return F.mse_loss(r_pred, r_ref) * w_mse


# ---------------------------
# 2) 频谱斜率/截距（尺寸分布）
# ---------------------------

def spectral_slope_intercept_loss(I_ref: torch.Tensor, I_pred: torch.Tensor,
                                  frac_range: Tuple[float, float] = (0.1, 0.5),
                                  window=True, w_slope=1.0, w_intercept=1.0):
    """
    在选定频段内拟合 log谱 ~ a*r + b，匹配 (a,b)
    frac_range: 使用径向频率的比例区间，例如(0.1, 0.5)
    """
    ps_ref = _fft_power_spectrum(I_ref, window=window)
    ps_pred = _fft_power_spectrum(I_pred, window=window)
    r_ref = _radial_average(ps_ref)  # [B,R]
    r_pred = _radial_average(ps_pred)

    B, R = r_ref.shape
    rmin = int(R * frac_range[0])
    rmax = max(rmin + 4, int(R * frac_range[1]))
    rr = torch.arange(R, device=r_ref.device).float().unsqueeze(0).expand(B, -1)  # [B,R]

    x = rr[:, rmin:rmax]
    y_ref = r_ref[:, rmin:rmax]
    y_pred = r_pred[:, rmin:rmax]

    a_ref, b_ref = _fit_line_ls(x, y_ref)
    a_pred, b_pred = _fit_line_ls(x, y_pred)

    loss = w_slope * F.l1_loss(a_pred, a_ref) + w_intercept * F.l1_loss(b_pred, b_ref)
    return loss


# ---------------------------
# 3) 自相关/相关长度（轴向/横向）
# ---------------------------

def autocorr_corr_length_loss(I_ref: torch.Tensor, I_pred: torch.Tensor, w_axial=1.0, w_lateral=1.0):
    """
    对自相关函数在轴向/横向的半功率宽度进行匹配
    """
    R_ref = _autocorr2d(I_ref)  # [B,1,H,W]
    R_pred = _autocorr2d(I_pred)

    # 取中心行/列
    _, _, H, W = R_ref.shape
    cy, cx = H // 2, W // 2

    axial_ref = R_ref[:, 0, :, cx]        # [B,H]
    axial_pred = R_pred[:, 0, :, cx]
    lat_ref = R_ref[:, 0, cy, :]          # [B,W]
    lat_pred = R_pred[:, 0, cy, :]

    Lz_ref = _half_power_width(axial_ref)
    Lz_pred = _half_power_width(axial_pred)
    Lx_ref = _half_power_width(lat_ref)
    Lx_pred = _half_power_width(lat_pred)

    return w_axial * F.l1_loss(Lz_pred, Lz_ref) + w_lateral * F.l1_loss(Lx_pred, Lx_ref)


# ---------------------------
# 4) 包络统计（Nakagami/Rayleigh）
# ---------------------------

def nakagami_loss(envelope_ref: torch.Tensor, envelope_pred: torch.Tensor,
                  win: int = 32, stride: int = 16, w_m=1.0, w_omega=0.5, eps=1e-6):
    """
    使用方法矩估计 Nakagami (m, Omega) 并匹配
    建议传入包络图（若为B-mode，先反log到近似包络）
    """
    def _moments_nakagami(E):
        B, _, H, W = E.shape
        patches = F.unfold(E, kernel_size=win, stride=stride)  # [B, win*win, N]
        A2 = (patches ** 2).mean(dim=1)                       # [B,N]
        A4 = (patches ** 4).mean(dim=1)
        varA2 = (A4 - A2 ** 2).clamp_min(eps)
        m = (A2 ** 2 / varA2).clamp_min(eps)                  # Nakagami m
        Omega = A2
        return m, Omega

    m_r, Om_r = _moments_nakagami(envelope_ref)
    m_p, Om_p = _moments_nakagami(envelope_pred)
    return w_m * F.l1_loss(m_p, m_r) + w_omega * F.l1_loss(Om_p, Om_r)


# ---------------------------
# 5) 轴向衰减/阴影
# ---------------------------

def axial_attenuation_loss(envelope_ref: torch.Tensor, envelope_pred: torch.Tensor,
                           depth_axis: int = 2, z_min_frac: float = 0.2, z_max_frac: float = 0.8,
                           w_alpha=1.0, eps=1e-6):
    """
    在每条 A-line 上拟合 log(包络) ~ -alpha*z + c，匹配 alpha（可反映骨后阴影/衰减）
    envelope_*: [B,1,H,W]  -- 包络域更合理；若为B-mode可先反log到包络近似
    depth_axis: 2 表示 H 方向为深度（常见情况）
    """
    # 将尺度调整到便于线性回归的 log 域
    E_ref = envelope_ref.clamp_min(eps)
    E_pred = envelope_pred.clamp_min(eps)
    L_ref = torch.log(E_ref)
    L_pred = torch.log(E_pred)

    # 如果深度轴不是 H，则换轴到 H
    if depth_axis != 2:
        if depth_axis == 3:
            L_ref = L_ref.transpose(2, 3)
            L_pred = L_pred.transpose(2, 3)
        else:
            raise ValueError("depth_axis must be 2 (H) or 3 (W).")

    B, _, H, W = L_ref.shape
    zmin = int(H * z_min_frac)
    zmax = max(zmin + 8, int(H * z_max_frac))
    Kz = zmax - zmin

    # —— 正确的每列A-line回归设计矩阵（只含深度维Kz）——
    z_vec = torch.arange(zmin, zmax, device=L_ref.device).float()          # [Kz]
    X = torch.stack([z_vec, torch.ones_like(z_vec)], dim=1)                # [Kz,2]
    XtX = X.t() @ X
    X_pinv = torch.linalg.inv(XtX + 1e-6 * torch.eye(2, device=L_ref.device)) @ X.t()  # [2,Kz]

    def _fit_alpha(L):
        # 取 ROI: [B, Kz, W]
        Y = L[:, 0, zmin:zmax, :]                     # [B,Kz,W]
        # 变成 [B*W, Kz]，每行是一条A-line的纵向样本
        Y = Y.permute(0, 2, 1).contiguous().view(B * W, Kz)  # [B*W, Kz]
        # 线性回归解：theta = (X^T X)^{-1} X^T y ；向量化到所有A-line
        theta = (X_pinv @ Y.t()).t()                 # [B*W, 2]  -> 每条线的[a, c]
        a = theta[:, 0].view(B, W)                   # [B,W]
        alpha = (-a)                                  # alpha ~ -a
        return alpha

    alpha_r = _fit_alpha(L_ref)  # [B,W]
    alpha_p = _fit_alpha(L_pred) # [B,W]
    return w_alpha * F.l1_loss(alpha_p, alpha_r)



# ---------------------------
# 6) 散射“风格”Gram损失（物理滤波器）
# ---------------------------

class PhysicsAwareGramLoss(nn.Module):
    def __init__(self,
                 scales: List[int] = (7, 11, 15),
                 thetas: List[float] = (0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6),
                 sigma: float = 2.0,
                 w_gram: float = 1.0):
        super().__init__()
        # 注册为buffer，训练中不更新
        k = _gabor_kernels(list(scales), list(thetas), sigma=sigma)  # [N,1,kh,kw]
        self.register_buffer("kernels", k)
        self.w_gram = w_gram

    def forward(self, I_ref: torch.Tensor, I_pred: torch.Tensor):
        # 用固定核提取散射纹理特征
        F_ref = F.conv2d(I_ref, self.kernels, padding='same')
        F_pred = F.conv2d(I_pred, self.kernels, padding='same')
        G_ref = _gram_matrix(F_ref)  # [B,N,N]
        G_pred = _gram_matrix(F_pred)
        return self.w_gram * F.mse_loss(G_pred, G_ref)


# ---------------------------
# 组合器：一站式Physics-Guided纹理/散射损失
# ---------------------------

class PhysicsGuidedTextureLoss(nn.Module):
    def __init__(self,
                 w_radial_psd=1.0,
                 w_slope=0.2, w_intercept=0.2,
                 w_auto_axial=0.5, w_auto_lat=0.5,
                 w_nkg_m=0.5, w_nkg_omega=0.25,
                 w_alpha=0.2,
                 w_gram=0.2,
                 use_window=True,
                 slope_range=(0.1, 0.5),
                 gram_scales=(7, 11, 15),
                 gram_thetas=(0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6),
                 gram_sigma=2.0,
                 nkg_win=32, nkg_stride=16):
        super().__init__()
        self.w_radial_psd = w_radial_psd
        self.w_slope = w_slope
        self.w_intercept = w_intercept
        self.w_auto_axial = w_auto_axial
        self.w_auto_lat = w_auto_lat
        self.w_nkg_m = w_nkg_m
        self.w_nkg_omega = w_nkg_omega
        self.w_alpha = w_alpha
        self.use_window = use_window
        self.slope_range = slope_range
        self.nkg_win = nkg_win
        self.nkg_stride = nkg_stride

        self.gram_loss = PhysicsAwareGramLoss(
            scales=list(gram_scales),
            thetas=list(gram_thetas),
            sigma=gram_sigma,
            w_gram=w_gram
        )

    @staticmethod
    def _to_envelope_if_needed(x: torch.Tensor, assume_bmode: bool):
        """
        如果传入的是B-mode（log压缩），这里给一个保守近似把它映射回“伪包络”。
        更理想：你在外部就提供包络图，或用已知动态范围做更精确的反变换。
        """
        if not assume_bmode:
            return x
    
        # 归一 & 去量纲，避免溢出
        # 用 amax 支持多维
        xmax = torch.amax(x, dim=(2, 3), keepdim=True)
        x0 = x - xmax                      # 让最大值在0附近
        k = 2.0                            # 可调系数（1~3常见）
        env = torch.exp(k * x0).clamp_min(1e-8)
        return env


    def forward(self,
                I_ref: torch.Tensor,
                I_pred: torch.Tensor,
                inputs_are_bmode: bool = True,
                return_dict: bool = False):
        """
        I_ref, I_pred: [B,1,H,W]
        inputs_are_bmode: 若为True，则在计算 Nakagami/Attenuation 时内部做近似反log到包络
        return_dict: 如果True，则返回 (loss_total, dict_of_sub_losses)
        """
        loss_dict = {}
        loss_total = 0.0
    
        # 1) Radial PSD
        if self.w_radial_psd > 0:
            l = radial_spectrum_loss(I_ref, I_pred, window=self.use_window, w_mse=1.0)
            loss_dict['radial_psd'] = l.item()
            loss_total += self.w_radial_psd * l
    
        # 2) Spectral slope/intercept
        if (self.w_slope > 0) or (self.w_intercept > 0):
            l = spectral_slope_intercept_loss(
                I_ref, I_pred, frac_range=self.slope_range,
                window=self.use_window,
                w_slope=self.w_slope, w_intercept=self.w_intercept
            )
            loss_dict['spectral_slope_intercept'] = l.item()
            loss_total += l
    
        # 3) Autocorr corr-length
        if (self.w_auto_axial > 0) or (self.w_auto_lat > 0):
            l = autocorr_corr_length_loss(
                I_ref, I_pred, w_axial=self.w_auto_axial, w_lateral=self.w_auto_lat
            )
            loss_dict['autocorr_corr_length'] = l.item()
            loss_total += l
    
        # 4) Nakagami
        if (self.w_nkg_m > 0) or (self.w_nkg_omega > 0):
            E_ref = self._to_envelope_if_needed(I_ref, inputs_are_bmode)
            E_pred = self._to_envelope_if_needed(I_pred, inputs_are_bmode)
            l = nakagami_loss(E_ref, E_pred,
                              win=self.nkg_win, stride=self.nkg_stride,
                              w_m=self.w_nkg_m, w_omega=self.w_nkg_omega)
            loss_dict['nakagami'] = l.item()
            loss_total += l
    
        # 5) Axial attenuation
        if self.w_alpha > 0:
            E_ref = self._to_envelope_if_needed(I_ref, inputs_are_bmode)
            E_pred = self._to_envelope_if_needed(I_pred, inputs_are_bmode)
            l = axial_attenuation_loss(E_ref, E_pred,
                                       depth_axis=2, z_min_frac=0.2, z_max_frac=0.8,
                                       w_alpha=self.w_alpha)
            loss_dict['attenuation'] = l.item()
            loss_total += l
    
        # 6) Physics-aware Gram
        l = self.gram_loss(I_ref, I_pred)
        loss_dict['gram'] = l.item()
        loss_total += l
    
        if return_dict:
            return loss_total, loss_dict
        else:
            return loss_total

class PhysicsGuidedTextureLossNormalized(nn.Module):
    """
    归一化/相对化版本：
    - Radial PSD:  相对误差  ||Δ||^2 / (||ref||^2 + eps)
    - Spectral Slope/Intercept:  |Δ| / (|ref| + eps)
    - Autocorr Corr-Length:  |ΔLz|/H + |ΔLx|/W
    - Nakagami:  MAE( |Δm|/(m_ref+eps) ) + λ * MAE( |ΔΩ|/(Ω_ref+eps) )
    - Attenuation α:  MAE( |Δα| / (|α_ref| + att_rel_eps) )
    - Gram:  ||ΔG||_F / (||G_ref||_F + eps)
    """
    def __init__(self,
                 w_radial_psd=1.0,
                 w_slope=0.2, w_intercept=0.2,
                 w_auto_axial=0.5, w_auto_lat=0.5,
                 w_nkg_m=0.5, w_nkg_omega=0.25,
                 w_alpha=0.2,
                 w_gram=0.2,
                 use_window=True,
                 slope_range=(0.1, 0.5),
                 gram_scales=(7, 11, 15),
                 gram_thetas=(0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6),
                 gram_sigma=2.0,
                 nkg_win=32, nkg_stride=16,
                 att_rel_eps=1e-3,       # 防止 α_ref 过小导致发散
                 eps=1e-6):
        super().__init__()
        self.w_radial_psd = w_radial_psd
        self.w_slope = w_slope
        self.w_intercept = w_intercept
        self.w_auto_axial = w_auto_axial
        self.w_auto_lat = w_auto_lat
        self.w_nkg_m = w_nkg_m
        self.w_nkg_omega = w_nkg_omega
        self.w_alpha = w_alpha
        self.w_gram = w_gram
        self.use_window = use_window
        self.slope_range = slope_range
        self.nkg_win = nkg_win
        self.nkg_stride = nkg_stride
        self.att_rel_eps = att_rel_eps
        self.eps = eps

        # 预备Gram核
        k = _gabor_kernels(list(gram_scales), list(gram_thetas), sigma=gram_sigma)  # [N,1,K,K] （已pad统一尺寸）
        self.register_buffer("gram_kernels", k)

    @staticmethod
    def _to_envelope_if_needed(x: torch.Tensor, assume_bmode: bool):
        if not assume_bmode:
            return x
        xmax = torch.amax(x, dim=(2, 3), keepdim=True)
        x0 = x - xmax
        k = 2.0
        env = torch.exp(k * x0).clamp_min(1e-8)
        return env

    # ---------- 各子项的“相对/归一化”实现 ----------

    def _loss_radial_psd_rel(self, I_ref, I_pred):
        ps_ref = _fft_power_spectrum(I_ref, window=self.use_window)  # [B,1,H,W]
        ps_pred = _fft_power_spectrum(I_pred, window=self.use_window)
        r_ref = _radial_average(ps_ref)  # [B,R]
        r_pred = _radial_average(ps_pred)
        # 不再做标准化；直接做相对误差
        num = torch.mean((r_pred - r_ref) ** 2, dim=1)           # [B]
        den = torch.mean((r_ref) ** 2, dim=1).clamp_min(self.eps)
        rel = (num / den).mean()
        return rel

    def _loss_slope_intercept_rel(self, I_ref, I_pred):
        ps_ref = _fft_power_spectrum(I_ref, window=self.use_window)
        ps_pred = _fft_power_spectrum(I_pred, window=self.use_window)
        r_ref = _radial_average(ps_ref)  # [B,R]
        r_pred = _radial_average(ps_pred)
        B, R = r_ref.shape
        rmin = int(R * self.slope_range[0])
        rmax = max(rmin + 4, int(R * self.slope_range[1]))
        rr = torch.arange(R, device=r_ref.device).float().unsqueeze(0).expand(B, -1)
        x = rr[:, rmin:rmax]
        y_ref = r_ref[:, rmin:rmax]
        y_pred = r_pred[:, rmin:rmax]
        a_ref, b_ref = _fit_line_ls(x, y_ref)
        a_pred, b_pred = _fit_line_ls(x, y_pred)
        slope_rel = torch.mean(torch.abs(a_pred - a_ref) / (torch.abs(a_ref) + self.eps))
        intcpt_rel = torch.mean(torch.abs(b_pred - b_ref) / (torch.abs(b_ref) + self.eps))
        return slope_rel, intcpt_rel

    def _loss_autocorr_corrlen_rel(self, I_ref, I_pred):
        R_ref = _autocorr2d(I_ref)  # [B,1,H,W]
        R_pred = _autocorr2d(I_pred)
        _, _, H, W = R_ref.shape
        cy, cx = H // 2, W // 2
        axial_ref = R_ref[:, 0, :, cx]  # [B,H]
        axial_pred = R_pred[:, 0, :, cx]
        lat_ref = R_ref[:, 0, cy, :]    # [B,W]
        lat_pred = R_pred[:, 0, cy, :]
        Lz_ref = _half_power_width(axial_ref)  # in px
        Lz_pred = _half_power_width(axial_pred)
        Lx_ref = _half_power_width(lat_ref)
        Lx_pred = _half_power_width(lat_pred)
        dz = torch.abs(Lz_pred - Lz_ref) / (H + self.eps)  # [B]
        dx = torch.abs(Lx_pred - Lx_ref) / (W + self.eps)  # [B]
        return dz.mean(), dx.mean()

    def _loss_nakagami_rel(self, E_ref, E_pred):
        B, _, H, W = E_ref.shape
        patches_r = F.unfold(E_ref, kernel_size=self.nkg_win, stride=self.nkg_stride)  # [B, win*win, N]
        patches_p = F.unfold(E_pred, kernel_size=self.nkg_win, stride=self.nkg_stride)
        A2_r = (patches_r ** 2).mean(dim=1)         # [B,N]
        A4_r = (patches_r ** 4).mean(dim=1)
        varA2_r = (A4_r - A2_r ** 2).clamp_min(self.eps)
        m_r = (A2_r ** 2 / varA2_r).clamp_min(self.eps)
        Om_r = A2_r.clamp_min(self.eps)

        A2_p = (patches_p ** 2).mean(dim=1)
        A4_p = (patches_p ** 4).mean(dim=1)
        varA2_p = (A4_p - A2_p ** 2).clamp_min(self.eps)
        m_p = (A2_p ** 2 / varA2_p).clamp_min(self.eps)
        Om_p = A2_p.clamp_min(self.eps)

        m_rel = torch.mean(torch.abs(m_p - m_r) / (m_r + self.eps))
        Om_rel = torch.mean(torch.abs(Om_p - Om_r) / (Om_r + self.eps))
        return m_rel, Om_rel

    def _loss_att_rel(self, E_ref, E_pred):
        # 拟合 log(包络) ~ a*z + c；alpha = -a
        def fit_alpha(L):
            B, _, H, W = L.shape
            zmin = int(H * 0.2); zmax = max(zmin + 8, int(H * 0.8))
            Kz = zmax - zmin
            z_vec = torch.arange(zmin, zmax, device=L.device).float()
            X = torch.stack([z_vec, torch.ones_like(z_vec)], dim=1)      # [Kz,2]
            XtX = X.t() @ X
            X_pinv = torch.linalg.inv(XtX + 1e-6 * torch.eye(2, device=L.device)) @ X.t()  # [2,Kz]
            Y = L[:, 0, zmin:zmax, :]                                    # [B,Kz,W]
            Y = Y.permute(0, 2, 1).contiguous().view(B * W, Kz)          # [B*W,Kz]
            theta = (X_pinv @ Y.t()).t()                                 # [B*W,2]
            a = theta[:, 0].view(B, W)
            alpha = (-a)
            return alpha

        Lr = torch.log(E_ref.clamp_min(self.eps))
        Lp = torch.log(E_pred.clamp_min(self.eps))
        alpha_r = fit_alpha(Lr)   # [B,W]
        alpha_p = fit_alpha(Lp)   # [B,W]
        denom = alpha_r.abs() + self.att_rel_eps
        rel = torch.mean(torch.abs(alpha_p - alpha_r) / denom)
        return rel

    def _loss_gram_rel(self, I_ref, I_pred):
        pad = self.gram_kernels.shape[-1] // 2
        F_ref = F.conv2d(I_ref, self.gram_kernels, padding=pad)
        F_pred = F.conv2d(I_pred, self.gram_kernels, padding=pad)
        G_ref = _gram_matrix(F_ref)  # [B,N,N]
        G_pred = _gram_matrix(F_pred)
        num = torch.linalg.norm(G_pred - G_ref, dim=(1, 2))      # [B]
        den = torch.linalg.norm(G_ref, dim=(1, 2)).clamp_min(self.eps)
        rel = (num / den).mean()
        return rel

    # ---------- 对外接口 ----------

    def forward(self,
                I_ref: torch.Tensor,
                I_pred: torch.Tensor,
                inputs_are_bmode: bool = True,
                return_dict: bool = True):
        """
        返回 (total_loss, dict_of_sub_losses) ；若 return_dict=False，则仅返回标量
        """
        loss_dict = {}
        total = 0.0

        # 1) Radial PSD (relative)
        if self.w_radial_psd > 0:
            l = self._loss_radial_psd_rel(I_ref, I_pred)
            loss_dict['radial_psd_rel'] = l.item()
            total = total + self.w_radial_psd * l

        # 2) Spectral slope & intercept (relative)
        if (self.w_slope > 0) or (self.w_intercept > 0):
            l_s, l_b = self._loss_slope_intercept_rel(I_ref, I_pred)
            loss_dict['spectral_slope_rel'] = l_s.item()
            loss_dict['spectral_intercept_rel'] = l_b.item()
            total = total + self.w_slope * l_s + self.w_intercept * l_b

        # 3) Autocorr Corr-Length (normalized by size)
        if (self.w_auto_axial > 0) or (self.w_auto_lat > 0):
            l_z, l_x = self._loss_autocorr_corrlen_rel(I_ref, I_pred)
            loss_dict['autocorr_axial_rel'] = l_z.item()
            loss_dict['autocorr_lateral_rel'] = l_x.item()
            total = total + self.w_auto_axial * l_z + self.w_auto_lat * l_x

        # 4) Nakagami (relative)
        if (self.w_nkg_m > 0) or (self.w_nkg_omega > 0):
            E_ref = self._to_envelope_if_needed(I_ref, inputs_are_bmode)
            E_pred = self._to_envelope_if_needed(I_pred, inputs_are_bmode)
            l_m, l_Om = self._loss_nakagami_rel(E_ref, E_pred)
            loss_dict['nakagami_m_rel'] = l_m.item()
            loss_dict['nakagami_Omega_rel'] = l_Om.item()
            total = total + self.w_nkg_m * l_m + self.w_nkg_omega * l_Om

        # 5) Axial attenuation (relative)
        if self.w_alpha > 0:
            E_ref = self._to_envelope_if_needed(I_ref, inputs_are_bmode)
            E_pred = self._to_envelope_if_needed(I_pred, inputs_are_bmode)
            l = self._loss_att_rel(E_ref, E_pred)
            loss_dict['attenuation_rel'] = l.item()
            total = total + self.w_alpha * l

        # 6) Gram (relative)
        if self.w_gram > 0:
            l = self._loss_gram_rel(I_ref, I_pred)
            loss_dict['gram_rel'] = l.item()
            total = total + self.w_gram * l

        if return_dict:
            return total, loss_dict
        else:
            return total
