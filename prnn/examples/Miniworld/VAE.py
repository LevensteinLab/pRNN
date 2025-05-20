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

from ratinabox.utils import get_angle
from ratinabox.Agent import Agent


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate: float, net_config: tuple, in_channels: int, latent_dim: int):
        super().__init__()
        self._learning_rate = learning_rate
        activation = nn.ReLU()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        n_channels, kernel_sizes, strides, paddings, output_paddings = net_config
        in_channels = self.in_channels

        ###########################
        # 1. Build Encoder
        ###########################
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
            modules.append(activation)
            in_channels = n_channels[i]

        # Flatten and Linear encoder
        modules.append(nn.Flatten())
        modules.append(nn.Linear(n_channels[-1] * 16 * 16, self.latent_dim))

        self.encoder = nn.Sequential(*modules)

        ###########################
        # 2. Build Decoder
        ###########################
        modules = []

        n_channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()
        paddings.reverse()
        n_channels.append(self.in_channels)

        # Flatten and Linear encoder
        modules.append(nn.Flatten())
        decoder_lin = nn.Sequential(
            nn.Linear(self.latent_dim, n_channels[0] * 16 * 16),
            nn.Unflatten(dim=1, unflattened_size=(n_channels[0], 16, 16)),
            activation,
        )
        modules.append(decoder_lin)

        # reverse CNN
        for i in range(len(n_channels) - 1):

            modules.append(
                nn.ConvTranspose2d(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=output_paddings[i],
                ),
            )
            modules.append(activation)

        self.decoder = nn.Sequential(*modules)

        self.save_hyperparameters(ignore=["net_config"])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def generate(self, x):
        return self.forward(x)[0]

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)


def run_vae_experiment(config: DictConfig):
    original_cwd = hydra.utils.get_original_cwd()
    rat_data_module = RatDataModule(
        data_dir=os.path.abspath(original_cwd + config["hardware"]["smp_dataset_folder_path"]),
        config=config,
        batch_size=config["vae"]["train_batch_size"],
        num_workers=config["hardware"]["num_data_loader_workers"], #!!
        img_size=config["env"]["img_size"], #!!
    )

    ae = LitAutoEncoder(
        learning_rate=config["vae"]["learning_rate"],
        net_config=config["vae"]["net_config"].values(),
        in_channels=config["vae"]["in_channels"],
        latent_dim=config["vae"]["latent_dim"],
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        filename="rat_ae-{epoch:02d}-{train_loss:.6f}",
        save_top_k=3,
        mode="min",
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(
        max_steps=config["vae"]["max_steps"],
        max_epochs=config["vae"]["max_epochs"],
        callbacks=[checkpoint_callback],
        default_root_dir=original_cwd,
        logger=tb_logger,
        # profiler="simple",
    )
    trainer.fit(ae, datamodule=rat_data_module)


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


class MiniworldRandomAgent(Agent):        
    def __init__(self, Environment, params={
                                    "dt": 0.1,
                                    "speed_mean": 0.2,
                                    "thigmotaxis": 0.2,
                                    "wall_repel_distance": 0.2,
                                    }):
        
        super().__init__(Environment, params)
        self.reset()

    def update(self, dt=None, drift_velocity=None, drift_to_random_strength_ratio=1):
        super().update(dt, drift_velocity, drift_to_random_strength_ratio)
        self.history["speed"].append(
            np.linalg.norm(np.array(self.history["pos"][-1]) - np.array(self.history["pos"][-2]))
        )

        angle_now = get_angle(np.array(self.history["pos"][-1]) - np.array(self.history["pos"][-2]))
        angle_before = self.history["angle"][-1]
        if abs(angle_now - angle_before) > np.pi:
            if angle_now > angle_before:
                angle_now -= 2 * np.pi
            elif angle_now < angle_before:
                angle_before -= 2 * np.pi
        self.history["rotation"].append(angle_now - angle_before)
        self.history["angle"].append(angle_now)
        return


    def generateActionSequence(self, pos, direction, T=1000):
        self.pos = pos
        self.velocity = self.speed_std * np.array([np.cos(direction), np.sin(direction)])
        self.history["pos"] = [self.pos]
        self.history["vel"] = [self.velocity]
        self.history["speed"] = [np.linalg.norm(self.velocity)]
        self.history["angle"] = [get_angle(self.velocity)]

        for i in range(T):
            self.update()

        traj = np.vstack((np.array(self.history["speed"]) * 10, np.array(self.history["rotation"])))

        return traj[:, 1:]
    
    def reset(self):
        self.reset_history()
        self.initialise_position_and_velocity()
        self.history["t"] = [0]
        self.history["pos"] = [self.pos]
        self.history["vel"] = [self.velocity]
        self.history["rot_vel"] = [self.rotational_velocity]
        self.history["speed"] = [np.linalg.norm(self.velocity)]
        self.history["rotation"] = [0]
        self.history["angle"] = [get_angle(self.velocity)]

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