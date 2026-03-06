import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FrozenResNet18(nn.Module):
    """ResNet18 pretrained on ImageNet, frozen, used as a visual encoder.

    The FC head is removed; features are taken from the adaptive average pool
    (512-dim). Input frames should be (N, 3, H, W) float tensors in [0, 1].

    Args:
        latent_dim: output dimension. If None, uses raw 512-dim pool features.
                    Otherwise a linear projection is added.
    """

    def __init__(self, latent_dim=None):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Drop the FC head; keep everything up to and including avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())

        if latent_dim is not None:
            self.proj = nn.Linear(512, latent_dim)
            self.latent_dim = latent_dim
        else:
            self.proj = None
            self.latent_dim = 512

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        z = self.features(x)
        if self.proj is not None:
            z = self.proj(z)
        return z
