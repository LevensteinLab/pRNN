import os
import torch
import hydra

import numpy as np
import lightning.pytorch as pl

from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm
from omegaconf import DictConfig


class VAE(nn.Module):
    def __init__(self, learning_rate: float, net_config: tuple,
                 in_channels: int, latent_dim: int, kld_weight=0.005):
        super().__init__()
        self.learning_rate = learning_rate
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight

        n_channels, kernel_sizes, strides, paddings, output_paddings = net_config

        self.build_encoder(in_channels, n_channels, kernel_sizes, strides, paddings)

        self.build_decoder(n_channels, kernel_sizes, strides, paddings, output_paddings)

        self.initialize_weights()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def build_encoder(self, in_channels, n_channels, kernel_sizes, strides, paddings):
        modules = []

        # CNN
        for i in range(len(n_channels)):
            modules.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                )
            )
            modules.append(self.activation)
            in_channels = n_channels[i]

        # Flatten and assemble
        modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(n_channels[-1] * 16 * 16, self.latent_dim)
        self.fc_log_var = nn.Linear(n_channels[-1] * 16 * 16, self.latent_dim)

    def build_decoder(self, n_channels, kernel_sizes, strides, paddings, output_paddings):
        modules = []

        n_channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()
        paddings.reverse()
        n_channels.append(self.in_channels)

        # Linear layer to map latent space to feature space
        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, n_channels[0] * 16 * 16),
            nn.Unflatten(dim=1, unflattened_size=(n_channels[0], 16, 16)),
            self.activation,
        )

        # Reverse CNN
        for i in range(len(n_channels) - 1):
            modules.append(
                nn.ConvTranspose2d(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=output_paddings[i],
                )
            )
            modules.append(self.activation)

        self.decoder = nn.Sequential(*modules)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)  # Xavier initialization for convolutional layers
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # Xavier initialization for linear layers
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        z = self.decoder_input(z)
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var, z

    def compute_loss(self, x, x_hat, mu, log_var):
        recons_loss = F.mse_loss(x_hat, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        loss = recons_loss + self.kld_weight * kld_loss
        return {"loss": loss, "reconstruction loss": recons_loss, "kld loss": kld_loss}


class VarAutoEncoder(pl.LightningModule): # This is for VAE pretraining
    def __init__(self, learning_rate: float, net_config: tuple,
                 in_channels: int, latent_dim: int, kld_weight=0.005):
        super().__init__()
        self._learning_rate = learning_rate
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight

        n_channels, kernel_sizes, strides, paddings, output_paddings = net_config

        self.build_encoder(in_channels, n_channels, kernel_sizes, strides, paddings)

        self.build_decoder(n_channels, kernel_sizes, strides, paddings, output_paddings)

        self.initialize_weights()

        self.save_hyperparameters(ignore=["net_config"])

    def build_encoder(self, in_channels, n_channels, kernel_sizes, strides, paddings):
        modules = []

        # CNN
        for i in range(len(n_channels)):
            modules.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                )
            )
            modules.append(self.activation)
            in_channels = n_channels[i]

        # Flatten and assemble
        modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(n_channels[-1] * 16 * 16, self.latent_dim)
        self.fc_log_var = nn.Linear(n_channels[-1] * 16 * 16, self.latent_dim)

    def build_decoder(self, n_channels, kernel_sizes, strides, paddings, output_paddings):
        modules = []

        n_channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()
        paddings.reverse()
        n_channels.append(self.in_channels)

        # Linear layer to map latent space to feature space
        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, n_channels[0] * 16 * 16),
            nn.Unflatten(dim=1, unflattened_size=(n_channels[0], 16, 16)),
            self.activation,
        )

        # Reverse CNN
        for i in range(len(n_channels) - 1):
            modules.append(
                nn.ConvTranspose2d(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=output_paddings[i],
                )
            )
            modules.append(self.activation)

        self.decoder = nn.Sequential(*modules)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)  # Xavier initialization for convolutional layers
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # Xavier initialization for linear layers
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize biases to zero

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        z = self.decoder_input(z)
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self.forward(x)
        recons_loss = F.mse_loss(x_hat, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        loss = recons_loss + self.kld_weight*kld_loss
        self.log("train_loss", loss)
        self.log("reconstruction_loss", recons_loss)
        self.log("kl_divergence", kld_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)


class ResNetVAE(pl.LightningModule):
    """VAE with ResNet18 encoder backbone and the same configurable transposed-CNN decoder.

    ResNet18 (without its FC head) always produces 512-dim features via adaptive
    average pooling, regardless of input resolution.  The decoder mirrors the
    VarAutoEncoder design and is configured via net_config.

    Args:
        decoder_spatial: spatial side-length of the feature map at the decoder
            bottleneck.  Use 16 for 64×64 images, 8 for 32×32 images.
    """

    def __init__(self, learning_rate: float, net_config: tuple,
                 in_channels: int, latent_dim: int, kld_weight: float = 0.005,
                 decoder_spatial: int = 16):
        super().__init__()
        self._learning_rate = learning_rate
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.decoder_spatial = decoder_spatial

        n_channels, kernel_sizes, strides, paddings, output_paddings = net_config

        # --- Encoder: ResNet18 backbone (adaptive avgpool kept, fc removed) ---
        backbone = resnet18(weights=None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False # need to double check how many inchannels for miniworld images
            )
        # children: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.encoder = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        resnet_out_dim = 512

        self.fc_mu = nn.Linear(resnet_out_dim, latent_dim)
        self.fc_log_var = nn.Linear(resnet_out_dim, latent_dim)

        # --- Decoder: same transposed-CNN as VarAutoEncoder ---
        # Copy lists so reversing them doesn't mutate the caller's config
        self._build_decoder(
            list(n_channels), list(kernel_sizes), list(strides),
            list(paddings), list(output_paddings),
        )

        self.save_hyperparameters(ignore=["net_config"])

    def _build_decoder(self, n_channels, kernel_sizes, strides, paddings, output_paddings):
        n_channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()
        paddings.reverse()
        n_channels.append(self.in_channels)

        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, n_channels[0] * self.decoder_spatial ** 2),
            nn.Unflatten(1, (n_channels[0], self.decoder_spatial, self.decoder_spatial)),
            self.activation,
        )

        modules = []
        for i in range(len(n_channels) - 1):
            modules.append(
                nn.ConvTranspose2d(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=output_paddings[i],
                )
            )
            modules.append(self.activation)
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def compute_loss(self, x, x_hat, mu, log_var):
        recons_loss = F.mse_loss(x_hat, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        loss = recons_loss + self.kld_weight * kld_loss
        return {"loss": loss, "reconstruction loss": recons_loss, "kld loss": kld_loss}

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var, z = self.forward(x)
        recons_loss = F.mse_loss(x_hat, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        loss = recons_loss + self.kld_weight * kld_loss
        self.log("train_loss", loss)
        self.log("reconstruction_loss", recons_loss)
        self.log("kl_divergence", kld_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)


class ResNetAE(pl.LightningModule):
    """Deterministic autoencoder with a ResNet18 encoder and transposed-CNN decoder.

    Drops the variational bottleneck entirely: the encoder produces a single
    embedding vector (no mu/log_var split, no reparameterization, no KL term).
    The loss is pure MSE reconstruction.

    Args:
        decoder_spatial: spatial side-length at the decoder bottleneck
            (16 for 64×64 images, 8 for 32×32 images).
    """

    def __init__(self, learning_rate: float, net_config: tuple,
                 in_channels: int, latent_dim: int,
                 decoder_spatial: int = 16, pretrained: bool = True, projection: bool = False):
        super().__init__()
        self._learning_rate = learning_rate
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.decoder_spatial = decoder_spatial

        n_channels, kernel_sizes, strides, paddings, output_paddings = net_config

        if pretrained and projection:
            print("Note: Using pretrained ResNet encoder with a randomly initialized projection layer. Pretrained weights may be modified during training of the projection layer...")
        # --- Encoder: ResNet18 backbone (adaptive avgpool kept, fc removed) ---
        
        backbone = resnet18(weights='DEFAULT') if pretrained else resnet18(weights=None) #use pretrained ResNet weights? 

        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.encoder = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        if pretrained:
            self.encoder.requires_grad_(False)  # freeze backbone; only fc_proj/decoder train

        # Optional FC projection: 512 -> latent_dim (trained from scratch)
        if projection:
            self.fc_proj = nn.Linear(512, latent_dim)
            self.latent_dim = latent_dim
        else:
            self.fc_proj = None
            self.latent_dim = 512  # raw ResNet18 output

        # --- Decoder: same transposed-CNN as ResNetVAE ---
        self._build_decoder(
            list(n_channels), list(kernel_sizes), list(strides),
            list(paddings), list(output_paddings),
        )

        self.save_hyperparameters(ignore=["net_config"])

        # Only optimize trainable params (excludes frozen backbone)
        trainable = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=self._learning_rate)

    def _build_decoder(self, n_channels, kernel_sizes, strides, paddings, output_paddings):
        n_channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()
        paddings.reverse()
        n_channels.append(self.in_channels)

        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, n_channels[0] * self.decoder_spatial ** 2),
            nn.Unflatten(1, (n_channels[0], self.decoder_spatial, self.decoder_spatial)),
            self.activation,
        )

        modules = []
        for i in range(len(n_channels) - 1):
            modules.append(
                nn.ConvTranspose2d(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=output_paddings[i],
                )
            )
            modules.append(self.activation)
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        """Returns (z, None) to match the VAE (mu, log_var) unpacking in Shell.py."""
        z = self.encoder(x)
        if self.fc_proj is not None:
            z = self.fc_proj(z)
        return z, None

    def reparameterize(self, mu, log_var):
        """Identity — no stochastic sampling in a plain AE."""
        return mu

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        """Returns (x_hat, z, None, z) to match the ResNetVAE 4-tuple in Shell.py."""
        z, _ = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, None, z

    def compute_loss(self, x, x_hat, mu, log_var):
        """Pure reconstruction loss — mu/log_var args kept for API compatibility."""
        loss = F.mse_loss(x_hat, x)
        return {"loss": loss, "reconstruction loss": loss, "kld loss": torch.tensor(0.0)}

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, z, _, _ = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer


class RatDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, config: DictConfig, batch_size: int = 50, num_workers: int = 0, img_size: int = 64
    ):
        super().__init__()
        self._data_dir = data_dir
        self._config = config
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._img_size = img_size
        self._img_dataset = None

    def setup(self, stage: str):
        self._img_dataset = ImageFolder(root=self._data_dir, transform=ToTensor())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self._img_dataset,
            shuffle=True,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=True if self._num_workers > 0 else False,
        )

def run_random_walk(time: int, dataset_folder_path: str, n_traj: int, env, agent,
                    view='ego', save_traj=True):
    for i in range(n_traj):
        env.reset()
        agent.reset()

        trajdir = os.path.join(dataset_folder_path, str(i))
        imgdir = os.path.join(trajdir, "Images")
        os.makedirs(imgdir, exist_ok=True)

        pos = env.unwrapped.agent.pos
        direction = env.unwrapped.agent.dir

        traj = agent.generateActionSequence(np.array([pos[0] - 0.5, env.size - pos[2] + 0.5]) / 10,
                                            direction, time)

        for t in tqdm(range(traj.shape[1]), desc=f"Render Images for trajectory #{i}"):
            if view == 'ego':
                render = env.render()
            elif view == 'top':
                render = env.render_top_view()
            else:
                raise ValueError("view must be either 'ego' or 'top'")
            Image.fromarray(render).save(os.path.join(imgdir, f"{t}.png"))

            action = traj[:, t]
            obs, reward, termination, truncation, info = env.step(action)

            if termination or truncation:
                env.reset()

        if save_traj:
            np.save(os.path.join(trajdir, "act.npy"), traj)
            np.save(os.path.join(trajdir, "pos.npy"), pos)
            np.save(os.path.join(trajdir, "dir.npy"), direction)

    env.close()