"""
MobileNetV3-Small + U-Net — exact architecture matching the training checkpoints.

MobileNetV3-Small feature map dimensions (224×224 input, torchvision):
    features[ 0]: (B, 16, 112, 112)
    features[ 1]: (B, 16,  56,  56)  ← skip3 (16ch)
    features[ 2]: (B, 24,  28,  28)  ← skip2 (24ch)
    features[ 3]: (B, 24,  28,  28)
    features[ 4]: (B, 40,  14,  14)
    features[ 5]: (B, 40,  14,  14)
    features[ 6]: (B, 40,  14,  14)
    features[ 7]: (B, 48,  14,  14)
    features[ 8]: (B, 48,  14,  14)  ← skip1 (48ch)
    features[ 9]: (B, 96,   7,   7)
    features[10]: (B, 96,   7,   7)
    features[11]: (B, 96,   7,   7)
    features[12]: (B, 576,  7,   7)  ← bridge

Checkpoint key structure:
    encoder.backbone.*               — full MobileNetV3-Small model (incl. classifier)
    encoder.features.{0..12}.*       — backbone.features Sequential (same weights)
    decoder.upconv1                   ConvTranspose2d(576 → 48,  2×2)  bridge→14×14
    decoder.conv_block1               Conv(96  → 48)  after cat(48+48)
    decoder.upconv2                   ConvTranspose2d(48  → 24,  2×2)  14→28
    decoder.conv_block2               Conv(48  → 24)  after cat(24+24)
    decoder.upconv3                   ConvTranspose2d(24  → 16,  2×2)  28→56
    decoder.conv_block3               Conv(32  → 16)  after cat(16+16)
    decoder.upconv4                   ConvTranspose2d(16  → 16,  2×2)  56→112
    decoder.conv_block4               Conv(32  → 16)  after cat(16+16)
    decoder.upconv_final              ConvTranspose2d(16  → 16,  2×2)  112→224
    decoder.final_conv                Conv2d(16 → 1, 1×1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class MobileNetV3SmallEncoder(nn.Module):
    """
    Wraps MobileNetV3-Small exposing skip connections at four spatial scales.

    The training code stored:
        self.backbone = mobilenet_v3_small(...)       (full model with classifier)
        self.features = mobilenet_v3_small.features   (the Sequential only)

    Both attributes are stored here to exactly reproduce checkpoint key structure.

    Skip stages collected during forward pass:
        features[0]  → (B, 16, 112, 112)  skip4 — shallowest
        features[1]  → (B, 16,  56,  56)  skip3
        features[2]  → (B, 24,  28,  28)  skip2
        features[8]  → (B, 48,  14,  14)  skip1 — deepest skip
        features[12] → (B, 576,  7,   7)  bridge
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        # Full model stored as .backbone → produces encoder.backbone.* keys
        self.backbone = models.mobilenet_v3_small(weights=weights)
        # .features reference → produces encoder.features.* keys
        self.features = self.backbone.features

    def forward(self, x: torch.Tensor):
        skips = []
        out = x
        for i, layer in enumerate(self.backbone.features):
            out = layer(out)
            if i == 0:    # (B, 16, 112, 112) — skip4
                skips.append(out)
            elif i == 1:  # (B, 16,  56,  56) — skip3
                skips.append(out)
            elif i == 2:  # (B, 24,  28,  28) — skip2
                skips.append(out)
            elif i == 8:  # (B, 48,  14,  14) — skip1
                skips.append(out)
        bridge = out  # (B, 576, 7, 7) after features[12]
        # Return skips ordered shallowest → deepest: [skip4, skip3, skip2, skip1]
        return bridge, skips


# ---------------------------------------------------------------------------
# Decoder conv block
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Two Conv3×3 → BN → ReLU layers matching decoder.conv_block{1..4} keys.

    Sequential indices used by checkpoint:
        0 — Conv2d(in_ch,  out_ch, 3, padding=1)
        1 — BatchNorm2d(out_ch)
        2 — ReLU
        3 — Conv2d(out_ch, out_ch, 3, padding=1)
        4 — BatchNorm2d(out_ch)
        5 — ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class UNetDecoder(nn.Module):
    """
    U-Net decoder with ConvTranspose2d upsampling and skip-connection fusion.

    Channel flow (derived from checkpoint weight shapes):
        bridge (576, 7×7)
          → upconv1 → (48, 14×14)  + cat skip1(48) → 96  → conv_block1 → 48
          → upconv2 → (24, 28×28)  + cat skip2(24) → 48  → conv_block2 → 24
          → upconv3 → (16, 56×56)  + cat skip3(16) → 32  → conv_block3 → 16
          → upconv4 → (16, 112×112)+ cat skip4(16) → 32  → conv_block4 → 16
          → upconv_final → (16, 224×224)
          → final_conv → (1, 224×224)
    """

    def __init__(self):
        super().__init__()
        self.upconv1     = nn.ConvTranspose2d(576, 48, kernel_size=2, stride=2)
        self.conv_block1 = _conv_block(96, 48)    # 48(up) + 48(skip1) = 96

        self.upconv2     = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.conv_block2 = _conv_block(48, 24)    # 24(up) + 24(skip2) = 48

        self.upconv3     = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.conv_block3 = _conv_block(32, 16)    # 16(up) + 16(skip3) = 32

        self.upconv4     = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv_block4 = _conv_block(32, 16)    # 16(up) + 16(skip4) = 32

        self.upconv_final = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_conv   = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Resize x to match skip if needed (handles odd-size padding), then concat."""
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, bridge: torch.Tensor, skips: list) -> torch.Tensor:
        # skips = [skip4(16,112), skip3(16,56), skip2(24,28), skip1(48,14)]
        skip4, skip3, skip2, skip1 = skips

        x = self.upconv1(bridge)            # (B, 48, 14, 14)
        x = self._cat(x, skip1)            # (B, 96, 14, 14)
        x = self.conv_block1(x)            # (B, 48, 14, 14)

        x = self.upconv2(x)                # (B, 24, 28, 28)
        x = self._cat(x, skip2)            # (B, 48, 28, 28)
        x = self.conv_block2(x)            # (B, 24, 28, 28)

        x = self.upconv3(x)                # (B, 16, 56, 56)
        x = self._cat(x, skip3)            # (B, 32, 56, 56)
        x = self.conv_block3(x)            # (B, 16, 56, 56)

        x = self.upconv4(x)                # (B, 16, 112, 112)
        x = self._cat(x, skip4)            # (B, 32, 112, 112)
        x = self.conv_block4(x)            # (B, 16, 112, 112)

        x = self.upconv_final(x)           # (B, 16, 224, 224)
        x = self.final_conv(x)             # (B,  1, 224, 224)
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MobileNetV3UNet(nn.Module):
    """
    Few-shot skin lesion segmentation model used during training.

    Input  : (B, 3, 224, 224)
    Output : (B, 1, 224, 224) — raw logits; apply sigmoid for probability map
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = MobileNetV3SmallEncoder(pretrained=pretrained)
        self.decoder = UNetDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bridge, skips = self.encoder(x)
        return self.decoder(bridge, skips)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def build_model(pretrained: bool = False) -> MobileNetV3UNet:
    """Return a freshly initialised model (no checkpoint loaded)."""
    return MobileNetV3UNet(pretrained=pretrained)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """
    Load a .pth checkpoint and return:
        {
          "model" : MobileNetV3UNet set to eval() mode,
          "meta"  : dict (best_val_dice, best_val_iou, epoch, best_epoch)
        }

    Handles both checkpoint formats:
        Format A — torch.save(model.state_dict(), path)
        Format B — torch.save({"model_state_dict": ..., "epoch": ..., ...}, path)
    """
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = MobileNetV3UNet(pretrained=False)
    meta: dict = {}

    if isinstance(raw, dict) and "model_state_dict" in raw:
        model.load_state_dict(raw["model_state_dict"])
        for key in ("best_val_dice", "best_val_iou", "epoch", "best_epoch", "train_size"):
            val = raw.get(key)
            if val is not None:
                meta[key] = val
        if "best_epoch" not in meta and "epoch" in meta:
            meta["best_epoch"] = meta["epoch"]
    else:
        model.load_state_dict(raw)

    model.to(device)
    model.eval()
    return {"model": model, "meta": meta}
