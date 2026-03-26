import os
import argparse

import torch
import lightning.pytorch as pl

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

from prnn.environments.Miniworld.VAE import RatDataModule, VarAutoEncoder, ResNetVAE, ResNetAE


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train a visual encoder on Miniworld images")

    parser.add_argument("--data_dir", default="miniworld_data",
                        help="Path to image dataset folder (ImageFolder format)")
    parser.add_argument("--output_dir", default="encoder_output",
                        help="Directory for checkpoints, logs, and saved model")
    parser.add_argument("--model_type", default="resnetae", choices=["vae", "resnet", "resnetae"],
                        help="Model type (default: resnetae)")

    # Architecture
    parser.add_argument("--latent_dim", default=128, type=int)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--decoder_spatial", default=16, type=int,
                        help="Spatial side-length at decoder bottleneck: 16 for 64x64 images, 8 for 32x32")
    parser.add_argument("--pretrained", default=True, type=lambda x: x.lower() != 'false',
                        help="Use ImageNet pretrained ResNet18 backbone (resnetae only, default: True)")
    parser.add_argument("--projection", default=True, type=lambda x: x.lower() != 'false',
                        help="Add linear projection 512->latent_dim (resnetae only, default: True)")

    # Decoder net_config (lists, one value per layer)
    parser.add_argument("--n_channels", nargs="+", type=int, default=[64, 32],
                        help="Decoder channel sizes per layer (default: 64 32)")
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[4, 4])
    parser.add_argument("--strides", nargs="+", type=int, default=[2, 2])
    parser.add_argument("--paddings", nargs="+", type=int, default=[1, 1])
    parser.add_argument("--output_paddings", nargs="+", type=int, default=[0, 0])

    # Loss
    parser.add_argument("--kld_weight", default=0.005, type=float)

    # Training
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Max training steps; -1 means no limit (use max_epochs)")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    net_config = (
        args.n_channels,
        args.kernel_sizes,
        args.strides,
        args.paddings,
        args.output_paddings,
    )

    rat_data_module = RatDataModule(
        data_dir=args.data_dir,
        config=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print('Data module created')

    common_kwargs = dict(
        learning_rate=args.learning_rate,
        net_config=net_config,
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        kld_weight=args.kld_weight,
    )
    if args.model_type == "resnet":
        ae = ResNetVAE(**common_kwargs, decoder_spatial=args.decoder_spatial)
    elif args.model_type == "resnetae":
        ae = ResNetAE(
            learning_rate=args.learning_rate,
            net_config=net_config,
            in_channels=args.in_channels,
            latent_dim=args.latent_dim,
            decoder_spatial=args.decoder_spatial,
            pretrained=args.pretrained,
            projection=args.projection,
        )
    else:
        ae = VarAutoEncoder(**common_kwargs)
    print(f'{args.model_type} created')

    checkpoint_path = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="train_loss",
        filename="encoder-{epoch:02d}-{train_loss:.6f}",
        save_top_k=3,
        mode="min",
    )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.output_dir, 'logs'))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
    )
    print('Starting training...')
    trainer.fit(ae, datamodule=rat_data_module)
    print('Training finished')

    # Save full model for easy loading in trainNet.py (torch.load)
    save_path = os.path.join(args.output_dir, 'encoder.pt')
    torch.save(ae, save_path)
    print(f'Encoder saved to {save_path}')


if __name__ == "__main__":
    main()
