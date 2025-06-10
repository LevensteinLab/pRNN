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
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm
from omegaconf import DictConfig


class VarAutoEncoder(pl.LightningModule):
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

        pos = env.agent.pos
        direction = env.agent.dir

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

    env.close()