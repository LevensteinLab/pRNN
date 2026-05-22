import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Res18(nn.Module):
    """ResNet18 pretrained on ImageNet, frozen, used as a visual encoder.
    The FC head is removed; features are taken from the adaptive average pool
    (512-dim). An optional learned linear projection reduces to latent_dim
    (used downstream). The backbone (self.features) is always frozen. The
    projection (self.proj), if present, is frozen by Res18, with subclasses
    managing its requires_grad.
    
    Input: (N, 3, H, W) float tensor in [0, 1], canonical NCHW layout. This
    layout is provided by the Shell.

    Args:
        latent_dim: output dimension. If None, uses raw 512-dim pool features
                    and no projection layer.
        bias: Boolean to indicate whether self.proj has a learnable bias term
                    if present.
    """

    def __init__(self, latent_dim=None, bias=True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # remove the FC head and flatten to (N, 512)
        self.features = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())

        # backbone is frozen for all Res18 variants
        for p in self.features.parameters():
            p.requires_grad = False

        if latent_dim is not None:
            self.proj = nn.Linear(512, latent_dim, bias=bias)
            self.latent_dim = latent_dim
            # projection also frozen by default, but can be overridden by subclasses
            for p in self.proj.parameters():
                p.requires_grad = False
        else:
            self.proj = None
            self.latent_dim = 512

        self.name = 'res'

    def encode_latent(self, x):
        """x: (N, 3, H, W) in [0, 1]. Returns (N, latent_dim)."""
        z = self.features(x)
        if self.proj is not None:
            z = self.proj(z)
        return z
    
    def forward(self, x):
        return self.encode_latent(x)

class Res18Random(Res18):
    """ResNet18 + frozen random linear projection to a lower-dim latent.

    512-dim ResNet features reduced to latent_dim via an untrained linear
    projection with no bias (pure Wx). Weight init is PyTorch default (Kaiming
    uniform); other choices (Gaussian JL, orthogonal) may be implemented later

    Args:
        latent_dim: output dimension. Required.
    """

    def __init__(self, latent_dim):
        # Inherit frozen ResNet18 backbone and frozen projection layer from Res18.
        # Pass bias=False since a random bias is just a fixed offset
        super().__init__(latent_dim=latent_dim, bias=False)
        self.name = 'res_random'

class Res18AE(Res18):
    """
    Frozen ResNet18 backbone + learned projection + learned conv decoder.

    Trained as an autoencoder on natural images:
        image -> latent -> reconstructed image
    The projection is unfrozen so it can be trained alongside the decoder.
    The ResNet18 backbone stays frozen.

    During pretraining: forward(x) returns (x_hat, z) for the reconstruction
    loop. At inference (inside Shell): encode_latent(x) returns just z.

    After pretraining, call freeze() to lock the projection and decoder during
    pRNN training.

    Args:
        latent_dim: bottleneck dimension. Required.
        img_size: square input image side length. Must be a multiple of 16.
                    The decoder geometry is 4 stride-2 upsamples from
                    (img_size//16)**2 to img_size**2.
    """

    def __init__(self, latent_dim, img_size):
        # Inherit frozen ResNet128 backbone and projection layer from Res18.
        # bias=True since the bias here is learned during pretraining.
        super().__init__(latent_dim=latent_dim, bias=True)
        self.name = 'res_ae'
        self.img_size = img_size

        # unfreeze projection layer
        for p in self.proj.parameters():
            p.requires_grad = True

        # decoder geometry: 4 stride-2 upsamples from (s0 x s0) to (img_size x 
        # img_size)
        assert img_size % 16 == 0, (
            f'img_size {img_size} must be a multiple of 16 for this decoder geometry'
        )
        s0 = img_size // 16

        # linear layer maps the latent back up to a small feature map, then
        # upsample to original image size
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 128 * s0 * s0),
            nn.Unflatten(1, (128, s0, s0)),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(), # outputs in [0,1] to match get_visual's /255 normalization
        )

    def decode(self, z):
        """z: (N, latent_dim). Returns reconstructed image (N, 3, img_size, img_size)"""
        return self.decoder(self.decoder_input(z))
    
    def forward(self, x):
        """Pretraining path. Returns (x_hat, z)."""
        z = self.encode_latent(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def freeze(self):
        """Freeze projection and decoder. Call after loading pretrained weights."""
        for p in self.proj.parameters():
            p.requires_grad = False
        for p in self.decoder_input.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.eval()
    
