import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTBandLoss(nn.Module):
    """
    Stable HF reconstruction loss on a frequency band, with cached Hann window + band mask.

    Args:
        lo (float): lower normalized radius (0..~1) for the band
        hi (float): upper normalized radius (0..~1) for the band
        eps (float): numerical epsilon for magnitude/log1p
        use_smoothl1 (bool): SmoothL1 over log-magnitudes (recommended)
    """
    def __init__(self, lo: float = 0.25, hi: float = 0.55, eps: float = 1e-6, use_smoothl1: bool = True):
        super().__init__()
        self.lo = float(lo)
        self.hi = float(hi)
        self.eps = float(eps)
        self.use_smoothl1 = bool(use_smoothl1)

        # Buffers are created lazily; keep them non-persistent so checkpoints stay lean.
        self.register_buffer("hann_win", torch.empty(0), persistent=False)  # shape: (1,1,H,W)
        self.register_buffer("band_mask", torch.empty(0), persistent=False) # shape: (H, W//2+1)
        self._cached_hw = None  # (H, W)

    @torch.no_grad()
    def _build_buffers(self, H: int, W: int, device: torch.device):
        # Hann window
        wh = torch.hann_window(H, device=device, dtype=torch.float32).unsqueeze(1)
        ww = torch.hann_window(W, device=device, dtype=torch.float32).unsqueeze(0)
        hann = (wh @ ww).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # Band mask in normalized radius
        yy = torch.linspace(0, 1, H, device=device, dtype=torch.float32).unsqueeze(1)
        xx = torch.linspace(0, 1, W // 2 + 1, device=device, dtype=torch.float32).unsqueeze(0)
        r = torch.sqrt(xx**2 + yy**2)
        mask = ((r >= self.lo) & (r <= self.hi)).float()  # (H, W//2+1)

        # Assign to buffers
        self.hann_win.resize_(hann.shape).copy_(hann)
        self.band_mask.resize_(mask.shape).copy_(mask)
        self._cached_hw = (H, W)

    def _ensure_buffers(self, x: torch.Tensor):
        H, W = x.shape[-2:]
        device = x.device
        needs_shape = (self._cached_hw != (H, W))
        needs_device = (self.hann_win.device != device) or (self.band_mask.device != device)
        if needs_shape or needs_device or self.hann_win.numel() == 0 or self.band_mask.numel() == 0:
            self._build_buffers(H, W, device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: images in [-1, 1], shape (B, C, H, W)
        Returns scalar loss (fp32).
        """
        # Make sure our buffers match the input
        self._ensure_buffers(x)

        # Do the math in fp32 for stability (safe even under AMP/Accelerate)
        x32 = x.float()
        y32 = y.float()

        # Windowed FFT
        X = torch.fft.rfft2(x32 * self.hann_win, norm="ortho")
        Y = torch.fft.rfft2(y32 * self.hann_win, norm="ortho")

        # Stable magnitudes and log domain
        magX = torch.clamp(torch.abs(X), min=self.eps)
        magY = torch.clamp(torch.abs(Y), min=self.eps)
        logX = torch.log1p(magX)  # bounded slope near 0
        logY = torch.log1p(magY)

        if self.use_smoothl1:
            diff = F.smooth_l1_loss(logX, logY, reduction="none")
        else:
            diff = (logX - logY).abs()

        # Normalize by per-sample content scale (detach to prevent feedback loops)
        # Broadcast dims: (B,C,H,W//2+1) -> mean over spatial/freq dims
        denom = (logY.abs().mean(dim=(-2, -1), keepdim=True).detach() + 1.0)
        rel = diff / denom

        # Apply band mask (H, W//2+1) -> broadcast across B,C
        rel = rel * self.band_mask

        return rel.mean()
