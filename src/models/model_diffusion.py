import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import DA2Encoder
from .decoder_diffusion import DepthUNet


def _cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0]).float()


class DepthDiffusionModel(nn.Module):
    """
    DA2 encoder (frozen, multi-layer) + U-Net denoiser.

    Denoising happens at full spatial resolution (560×560) in log-depth
    space.  The U-Net cross-attends to DA2 features at its 35×35 bottleneck.

    Training:  x0 = log(gt_depth), add noise → predict x0, supervise with
               silog on exp(pred_x0).
    Inference: DDIM sampling, configurable steps.
    Region:    sample_region() crops + zooms image, runs DA2 + denoiser at
               higher effective resolution for a specific patch region.
    """

    def __init__(
        self,
        encoder_size: str = "base",
        img_size:     int = 560,
        patch_size:   int = 14,
        base_ch:      int = 32,
        t_dim:        int = 256,
        T:            int = 1000,
    ):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.T          = T

        self.encoder = DA2Encoder(size=encoder_size, img_size=img_size)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.denoiser = DepthUNet(
            context_dim=self.encoder.embed_dim,
            base_ch=base_ch,
            t_dim=t_dim,
        )

        acp = _cosine_schedule(T)
        self.register_buffer("alphas_cumprod",       acp)
        self.register_buffer("sqrt_acp",             acp.sqrt())
        self.register_buffer("sqrt_one_minus_acp",   (1 - acp).sqrt())

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """Returns predicted depth map (B, 1, H, W) for silog loss."""
        B      = images.shape[0]
        device = images.device

        with torch.no_grad():
            context = self.encoder(images)              # (B, N, D)

        x0    = torch.log(depths.clamp(min=1e-6))       # (B, 1, H, W) log-space
        t     = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(x0)

        sqrt_a = self.sqrt_acp[t].view(B, 1, 1, 1)
        sqrt_b = self.sqrt_one_minus_acp[t].view(B, 1, 1, 1)
        x_t    = sqrt_a * x0 + sqrt_b * noise           # (B, 1, H, W)

        pred_log = self.denoiser(x_t, context, t)        # (B, 1, H, W)
        return torch.exp(pred_log).clamp(min=1e-3)

    # ------------------------------------------------------------------
    # DDIM
    # ------------------------------------------------------------------

    def _ddim_step(
        self,
        x:       torch.Tensor,
        context: torch.Tensor,
        t_cur:   torch.Tensor,
        t_next:  torch.Tensor,
        B:       int,
    ) -> torch.Tensor:
        pred_x0    = self.denoiser(x, context, t_cur.expand(B))
        alpha_cur  = self.alphas_cumprod[t_cur]
        alpha_next = self.alphas_cumprod[t_next]
        pred_noise = (x - alpha_cur.sqrt() * pred_x0) / (1 - alpha_cur).sqrt()
        return alpha_next.sqrt() * pred_x0 + (1 - alpha_next).sqrt() * pred_noise

    @torch.no_grad()
    def sample(self, images: torch.Tensor, num_steps: int = 20) -> torch.Tensor:
        """Full-image DDIM sampling → (B, 1, H, W) depth."""
        B      = images.shape[0]
        device = images.device

        context   = self.encoder(images)
        x         = torch.randn(B, 1, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(self.T - 1, 0, num_steps, dtype=torch.long, device=device)

        for i in range(num_steps - 1):
            x = self._ddim_step(x, context, timesteps[i], timesteps[i + 1], B)

        pred_log = self.denoiser(x, context, timesteps[-1].expand(B))
        return torch.exp(pred_log).clamp(min=1e-3)

    # ------------------------------------------------------------------
    # region refinement
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_region(
        self,
        images:    torch.Tensor,
        x0_coarse: torch.Tensor,
        row_start: int,
        row_end:   int,
        col_start: int,
        col_end:   int,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Refine depth for a specific pixel region at the model's full resolution.

        Crops + zooms the image region to img_size × img_size, runs DA2 for
        richer features of that sub-region, then denoises starting from a
        slightly-noised version of the coarse prediction (SDEdit-style).

        Returns refined depth (B, 1, img_size, img_size) for the crop —
        caller is responsible for stitching back into the full map.
        """
        B      = images.shape[0]
        device = images.device

        # crop and zoom image region to full model resolution
        crop_img   = images[:, :, row_start:row_end, col_start:col_end]
        crop_img   = F.interpolate(crop_img, size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False)

        # higher-resolution DA2 features for this region
        context = self.encoder(crop_img)

        # start from noised coarse prediction (SDEdit: partial noise, not full)
        crop_depth = x0_coarse[:, :, row_start:row_end, col_start:col_end]
        crop_depth = F.interpolate(crop_depth, size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False)
        x0_log = torch.log(crop_depth.clamp(min=1e-6))

        # add noise up to t_start — refine from there, not from pure noise
        t_start   = self.T // 4
        t_tensor  = torch.tensor(t_start, device=device)
        noise     = torch.randn_like(x0_log)
        x = (self.sqrt_acp[t_tensor] * x0_log
             + self.sqrt_one_minus_acp[t_tensor] * noise)

        timesteps = torch.linspace(t_start, 0, num_steps, dtype=torch.long, device=device)
        for i in range(num_steps - 1):
            x = self._ddim_step(x, context, timesteps[i], timesteps[i + 1], B)

        pred_log = self.denoiser(x, context, timesteps[-1].expand(B))
        return torch.exp(pred_log).clamp(min=1e-3)
