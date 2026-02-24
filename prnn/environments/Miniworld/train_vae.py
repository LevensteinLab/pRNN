import os
import hydra

import lightning.pytorch as pl

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

from prnn.environments.Miniworld.VAE import RatDataModule, VarAutoEncoder

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config):
    folder_path = os.path.join(os.path.expandvars('${SLURM_TMPDIR}'), config['fm']['encoder_folder'])
    print(f"Folder path: {folder_path}")
    rat_data_module = RatDataModule(
        data_dir=os.path.join(os.path.expandvars('${SLURM_TMPDIR}'), 'Miniworld-LRoom-v1', 'data'),
        config=config,
        batch_size=config["encoder"]["train_batch_size"],
        num_workers=2,
        img_size=64,
    )
    print('Data module created')
    
    ae = VarAutoEncoder(
        learning_rate=config["encoder"]["learning_rate"],
        net_config=config["encoder"]["net_config"].values(),
        in_channels=config["encoder"]["in_channels"],
        latent_dim=config["encoder"]["latent_dim"],
        kld_weight=config["encoder"]["kld_weight"],
    )
    print('Autoencoder created')
    
    checkpoint_path = os.path.join(folder_path, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="train_loss",
        filename="rat_ae-{epoch:02d}-{train_loss:.6f}",
        save_top_k=3,
        mode="min",
    )
    print('Checkpoint callback created')
    
    logs_path = os.path.join(folder_path, 'logs')
    os.makedirs(logs_path, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(logs_path)
    print('TensorBoard logger created')

    trainer = pl.Trainer(
        max_steps=config["encoder"]["max_steps"],
        max_epochs=config["encoder"]["max_epochs"],
        default_root_dir=folder_path,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
    )
    print('Trainer created')
    print('Starting training...')

    trainer.fit(ae, datamodule=rat_data_module)
    print('Training finished')

if __name__ == "__main__":
    main()