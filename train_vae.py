import os
import hydra

import lightning.pytorch as pl

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

from prnn.examples.Miniworld.VAE import RatDataModule, VarAutoEncoder


folder_path = os.path.join(os.path.expandvars('${SLURM_TMPDIR}'), 'Miniworld')
print(f"Folder path: {folder_path}")

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config):
    rat_data_module = RatDataModule(
        data_dir=os.path.join(folder_path, 'data'),
        config=config,
        batch_size=config["vae"]["train_batch_size"],
        num_workers=2,
        img_size=64,
    )
    
    ae = VarAutoEncoder(
        learning_rate=config["vae"]["learning_rate"],
        net_config=config["vae"]["net_config"].values(),
        in_channels=config["vae"]["in_channels"],
        latent_dim=config["vae"]["latent_dim"],
        kld_weight=config["vae"]["kld_weight"],
    )
    
    checkpoint_path = os.path.join(folder_path, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="train_loss",
        filename="rat_ae-{epoch:02d}-{train_loss:.6f}",
        save_top_k=3,
        mode="min",
    )
    
    logs_path = os.path.join(folder_path, 'logs')
    os.makedirs(logs_path, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(logs_path)

    trainer = pl.Trainer(
        max_steps=config["vae"]["max_steps"],
        max_epochs=config["vae"]["max_epochs"],
        # max_epochs=1,
        default_root_dir=folder_path,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
    )

    trainer.fit(ae, datamodule=rat_data_module)

if __name__ == "__main__":
    main()