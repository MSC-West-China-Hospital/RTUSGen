import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from physics_guided_losses import PhysicsGuidedTextureLossNormalized



class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_phy', type=float, default=1.0,
                    help='weight for physics-guided texture loss (applied on paired_B for A->B).')
            parser.add_argument('--phy_bmode', action='store_true',
                    help='inputs for physics loss are B-mode (log-compressed). If envelope, omit this flag.')
            # 结构项权重（新加）
            parser.add_argument('--lambda_lowfreq', type=float, default=1.0, help='L1 on low-pass images (paired_B only)')
            parser.add_argument('--lambda_ms_ssim', type=float, default=0.5, help='MS-SSIM on paired_B (structure)')
            parser.add_argument('--lambda_grad',    type=float, default=0.3, help='Gradient L1 (Sobel) on paired_B')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [
            'D_A','G_A','cycle_A','idt_A','D_B','G_B','cycle_B','idt_B','ssim_A','ssim_B',
            # physics-guided (all sub-terms; logged as scalars)
            'phy_radial','phy_slope','phy_intcpt',
            'phy_auto_ax','phy_auto_lat',
            'phy_m','phy_Om','phy_att','phy_gram',
            'struct_lowfreq', 'struct_ms_ssim', 'struct_grad'
        ]

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        # 如果用了 paired 图像，添加它们（仅在 test 阶段也无妨）
        self.paired_A = None
        self.paired_B = None

        if hasattr(self, 'paired_B'):
            visual_names_A.append('paired_B')
            print('has paired_A*****************************')
        if hasattr(self, 'paired_A'):
            visual_names_B.append('paired_A')
        #加的ssim
        self.lambda_ssim = opt.lambda_ssim
        
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.device_id)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.device_id)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.device_id)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.device_id)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.phy_loss_module = PhysicsGuidedTextureLossNormalized(
            w_radial_psd=1.0,
            w_slope=0.2, w_intercept=0.2,
            w_auto_axial=0.5, w_auto_lat=0.5,
            w_nkg_m=0.5, w_nkg_omega=0.25,
            w_alpha=0.2,
            w_gram=0.2,
            use_window=True,
            slope_range=(0.1, 0.5),
            gram_scales=(7, 11, 15),
            gram_thetas=(0, 0.52, 1.05, 1.57, 2.09, 2.62),
            gram_sigma=2.0,
            nkg_win=32, nkg_stride=16,
            att_rel_eps=1e-3
            ).to(self.device)
            # default zeros for logging when no paired_B in the batch
            self.loss_phy_radial   = torch.tensor(0.0, device=self.device)
            self.loss_phy_slope    = torch.tensor(0.0, device=self.device)
            self.loss_phy_intcpt   = torch.tensor(0.0, device=self.device)
            self.loss_phy_auto_ax  = torch.tensor(0.0, device=self.device)
            self.loss_phy_auto_lat = torch.tensor(0.0, device=self.device)
            self.loss_phy_m        = torch.tensor(0.0, device=self.device)
            self.loss_phy_Om       = torch.tensor(0.0, device=self.device)
            self.loss_phy_att      = torch.tensor(0.0, device=self.device)
            self.loss_phy_gram     = torch.tensor(0.0, device=self.device)
            self.loss_struct_lowfreq = torch.tensor(0.0, device=self.device)
            self.loss_struct_ms_ssim = torch.tensor(0.0, device=self.device)
            self.loss_struct_grad    = torch.tensor(0.0, device=self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # 加入 paired 图（如果存在）
        if 'paired_B' in input and input['paired_B'] is not None:
            self.paired_B = input['paired_B'].to(self.device)
        if 'paired_A' in input and input['paired_A'] is not None:
            self.paired_A = input['paired_A'].to(self.device)

        # 新增部分：paired 结构监督图像（文件名对齐）
        paired_B = input.get('paired_B', None)
        paired_A = input.get('paired_A', None)
    
        if paired_B is not None:
            self.paired_B = paired_B.to(self.device)
        else:
            self.paired_B = None
    
        if paired_A is not None:
            self.paired_A = paired_A.to(self.device)
        else:
            self.paired_A = None

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        if hasattr(self, 'paired_B'):
            self.paired_B = self.paired_B  # 为了可视化，显式赋值
        if hasattr(self, 'paired_A'):
            self.paired_A = self.paired_A

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # SSIM loss between real_B and fake_B (structure alignment)
        def normalize_to_01(x):
            return (x + 1) / 2
        #SSIM loss 配对图
        # self.loss_ssim_B = (1 - ssim(normalize_to_01(self.fake_B), normalize_to_01(self.real_B))) * self.lambda_ssim
        # self.loss_ssim_A = (1 - ssim(normalize_to_01(self.fake_A), normalize_to_01(self.real_A))) * self.lambda_ssim

        # SSIM with paired B
        if self.paired_B is not None:
            fake_B_01 = (self.fake_B + 1) / 2
            paired_B_01 = (self.paired_B + 1) / 2
            self.loss_ssim_B = (1 - ssim(fake_B_01, paired_B_01)) * self.lambda_ssim
        else:
            self.loss_ssim_B = 0
        
        # SSIM with paired A
        if self.paired_A is not None:
            fake_A_01 = (self.fake_A + 1) / 2
            paired_A_01 = (self.paired_A + 1) / 2
            self.loss_ssim_A = (1 - ssim(fake_A_01, paired_A_01)) * self.lambda_ssim
        else:
            self.loss_ssim_A = 0

        # ===== Physics-guided losses (detailed sub-terms) on paired_B =====
        phy_total_for_bp = torch.tensor(0.0, device=self.device)  # 用于反传的总和（加权）
        # 先把日志项都置零，避免无配对时残留上一批的值
        self.loss_phy_radial   = torch.tensor(0.0, device=self.device)
        self.loss_phy_slope    = torch.tensor(0.0, device=self.device)
        self.loss_phy_intcpt   = torch.tensor(0.0, device=self.device)
        self.loss_phy_auto_ax  = torch.tensor(0.0, device=self.device)
        self.loss_phy_auto_lat = torch.tensor(0.0, device=self.device)
        self.loss_phy_m        = torch.tensor(0.0, device=self.device)
        self.loss_phy_Om       = torch.tensor(0.0, device=self.device)
        self.loss_phy_att      = torch.tensor(0.0, device=self.device)
        self.loss_phy_gram     = torch.tensor(0.0, device=self.device)
        
        if self.paired_B is not None:
            # 网络输出是 [-1,1]，统一到 [0,1] 再送物理loss
            fake_B_01   = (self.fake_B   + 1) / 2
            paired_B_01 = (self.paired_B + 1) / 2

            # ---- 关键修复：确保是单通道 [B,1,H,W] ----
            def to_single_channel(x):
                # 如果是 3 通道，则转灰度（均值或选取第1通道均可）
                if x.dim() == 4 and x.size(1) == 3:
                    x = x.mean(dim=1, keepdim=True)   # 或 x[:, :1, ...]
                elif x.dim() == 4 and x.size(1) > 1:
                    x = x[:, :1, ...]
                return x
        
            fake_B_01   = to_single_channel(fake_B_01)
            paired_B_01 = to_single_channel(paired_B_01)
            
            # 现在是 [B,1,H,W]，可以安全送进 physics loss
            phy_total, phy_sub = self.phy_loss_module(
                paired_B_01, fake_B_01,
                inputs_are_bmode=self.opt.phy_bmode,
                return_dict=True
            )
        
            # —— 将子项写入日志变量（float->tensor）
            def t(x):  # helper: float or tensor -> tensor on device
                if isinstance(x, torch.Tensor):
                    return x.to(self.device) if x.device != self.device else x
                else:
                    return torch.tensor(float(x), device=self.device)
        
            self.loss_phy_radial   = t(phy_sub.get('radial_psd_rel', 0.0))
            self.loss_phy_slope    = t(phy_sub.get('spectral_slope_rel', 0.0))
            self.loss_phy_intcpt   = t(phy_sub.get('spectral_intercept_rel', 0.0))
            self.loss_phy_auto_ax  = t(phy_sub.get('autocorr_axial_rel', 0.0))
            self.loss_phy_auto_lat = t(phy_sub.get('autocorr_lateral_rel', 0.0))
            self.loss_phy_m        = t(phy_sub.get('nakagami_m_rel', 0.0))
            self.loss_phy_Om       = t(phy_sub.get('nakagami_Omega_rel', 0.0))
            self.loss_phy_att      = t(phy_sub.get('attenuation_rel', 0.0))
            self.loss_phy_gram     = t(phy_sub.get('gram_rel', 0.0))
        
            # —— 反传用：把“子项 × 内部权重”加总，再乘 lambda_phy
            # （PhysicsGuidedTextureLossNormalized 已经在 forward 里按各 w_* 组合过了）
            phy_total_for_bp = self.opt.lambda_phy * phy_total
        else:
            phy_total_for_bp = torch.tensor(0.0, device=self.device)

        # ===== Structure-preserving losses (only when paired_B is available) =====
        loss_lowfreq = torch.tensor(0.0, device=self.device)
        loss_ms_ssim = torch.tensor(0.0, device=self.device)
        loss_grad    = torch.tensor(0.0, device=self.device)
        
        if self.paired_B is not None:
            # 归一到 [0,1] 并单通道
            def to_gray01(x):
                x01 = (x + 1) / 2
                if x01.size(1) > 1:
                    x01 = x01.mean(dim=1, keepdim=True)
                return x01
        
            fake_B_01   = to_gray01(self.fake_B)
            paired_B_01 = to_gray01(self.paired_B)
        
            # --- 低频 L1（结构对齐）：可以保留，核适中即可，避免过强平滑 ---
            def blur(x, k=15):
                return torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=1, padding=k//2)
            fake_lp   = blur(fake_B_01, k=15)
            paired_lp = blur(paired_B_01, k=15)
            loss_lowfreq = torch.nn.functional.l1_loss(fake_lp, paired_lp)
        
            # --- 真实 MS-SSIM（更抗 speckle），直接在原图上算 ---
            loss_ms_ssim = 1.0 - ms_ssim(fake_B_01, paired_B_01, data_range=1.0, kernel_size=11, sigma=1.5, levels=5)
        
            # --- 梯度 L1（边缘） ---
            sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=self.device).view(1,1,3,3)
            sobel_y = sobel_x.transpose(2,3)
            def grad(x):
                gx = torch.nn.functional.conv2d(x, sobel_x, padding=1)
                gy = torch.nn.functional.conv2d(x, sobel_y, padding=1)
                return torch.sqrt(gx*gx + gy*gy + 1e-6)
            loss_grad = torch.nn.functional.l1_loss(grad(fake_B_01), grad(paired_B_01))

        # 记录到可打印字段（保持 tensor 类型，外层会转 float 打印）
        self.loss_struct_lowfreq = loss_lowfreq
        self.loss_struct_ms_ssim = loss_ms_ssim
        self.loss_struct_grad    = loss_grad

        # 把结构项加到总损失（带权重）
        self.loss_struct = (self.opt.lambda_lowfreq * loss_lowfreq
                            + self.opt.lambda_ms_ssim * loss_ms_ssim
                            + self.opt.lambda_grad    * loss_grad)

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B+self.loss_ssim_A + self.loss_ssim_B + phy_total_for_bp + self.loss_struct
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        
import torch.nn.functional as F

def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Simplified SSIM implementation (no window, fixed Gaussian kernel).
    Assumes img1 and img2 are tensors with shape (N, C, H, W) and normalized to [0, 1].
    """
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def _gaussian_window(kernel_size=11, sigma=1.5, channels=1, device='cpu', dtype=torch.float32):
    # 1D gaussian
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1)/2
    g = torch.exp(-(coords**2) / (2*sigma*sigma))
    g = g / g.sum()
    # 2D separable
    g2d = g[:, None] @ g[None, :]
    g2d = g2d / g2d.sum()
    window = g2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return window

def _ssim_per_scale(img1, img2, window, C1, C2):
    # img1, img2: [B,1,H,W] in [0,1]
    mu1 = F.conv2d(img1, window, padding=window.shape[-1]//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window.shape[-1]//2, groups=img2.size(1))
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window.shape[-1]//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window.shape[-1]//2, groups=img2.size(1)) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding=window.shape[-1]//2, groups=img1.size(1)) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    cs_map   = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    return ssim_map.mean(dim=[1,2,3]), cs_map.mean(dim=[1,2,3])

def ms_ssim(img1, img2, data_range=1.0, kernel_size=11, sigma=1.5, weights=None, levels=5):
    """
    MS-SSIM for grayscale images. img1,img2: [B,1,H,W] in [0,1]
    weights: list/tuple of length=levels (default per Wang et al. 2003)
    """
    assert img1.shape == img2.shape and img1.dim()==4 and img1.size(1)==1, "ms_ssim expects [B,1,H,W]"
    device, dtype = img1.device, img1.dtype
    if weights is None:
        # standard weights for 5 scales
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]
    weights = torch.tensor(weights, device=device, dtype=dtype)
    weights = weights / weights.sum()

    C1 = (0.01*data_range)**2
    C2 = (0.03*data_range)**2
    window = _gaussian_window(kernel_size, sigma, channels=1, device=device, dtype=dtype)

    mssim_list = []
    mcs_list   = []
    im1, im2 = img1, img2
    for _ in range(levels-1):
        ssim_val, cs_val = _ssim_per_scale(im1, im2, window, C1, C2)
        mssim_list.append(ssim_val)
        mcs_list.append(cs_val)
        # downsample
        im1 = F.avg_pool2d(im1, kernel_size=2, stride=2, padding=0)
        im2 = F.avg_pool2d(im2, kernel_size=2, stride=2, padding=0)

    # last scale
    ssim_val, cs_val = _ssim_per_scale(im1, im2, window, C1, C2)
    mssim_list.append(ssim_val)
    mcs_list.append(cs_val)

    mcs_stack   = torch.stack(mcs_list[:-1], dim=0)  # [levels-1,B]
    mssim_stack = torch.stack(mssim_list[-1:], dim=0) # [1,B]

    # MS-SSIM = prod(cs_i^{w_i}) * ssim_L^{w_L}
    log_mcs   = (weights[:-1].unsqueeze(1) * torch.log(torch.clamp(mcs_stack, min=1e-6))).sum(dim=0)
    log_mssim = (weights[-1].unsqueeze(0) * torch.log(torch.clamp(mssim_stack.squeeze(0), min=1e-6)))
    ms_ssim_val = torch.exp(log_mcs + log_mssim)  # [B]
    return ms_ssim_val.mean()  # scalar