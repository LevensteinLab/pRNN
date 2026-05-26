"""
Pretrain Res18AE on a directory of natural images (e.g. Places365).

Trains the projection and decoder by image reconstruction (MSE). The ResNet18
backbone stays frozen.

The dataset is loaded with a small recursive image-finder so it works on any
directory structure. Labels are ignored.

Usage:
    python -m prnn.utils.encoder_pretraining \
        --train_dir /gpfs/radev/project/levenstein/shared/places365/data_256_standard \
        --val_dir /gpfs/radev/project/levenstein/shared/places365/val_256 \
        --output_dir /gpfs/radev/project/levenstein/ac3787/prnn_training/nets/encoders/res_ae \
        --img_size 96 --latent_dim 128 \
        --batch_size 64 --epochs 20 --lr 1e-3 \
        --wandb_project samevr_encoder

Outputs:
    checkpoint.pt: torch.save({state_dict, optimizer, epoch}), overwritten each epoch
    hparams.json: hyperparameters used for this run (img_size, latent_dim, etc.)
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from PIL import Image

import wandb
from prnn.utils.encoder import Res18AE

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

class ImageDirDataset(Dataset):
    """Recursively finds all images under root. Returns (image_tensor, 0)."""
    def __init__(self, root, transform=None):
        self.paths = sorted(
            p for p in Path(root).rglob('*')
            if p.suffix.lower() in IMG_EXTS
        )
        if len(self.paths)==0:
            raise RuntimeError(f'No images found under {root}.')
        self.transform = transform

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Data
    p.add_argument('--train_dir', type=str, required=True,
                   help='Directory of training images (any depth)')
    p.add_argument('--val_dir', type=str, required=True,
                   help='Directory of validation images (any depth)')
    p.add_argument('--num_workers', type=int, default=4)
    # Model
    p.add_argument('--latent_dim', type=int, default=128)
    p.add_argument('--img_size', type=int, default=96,
                   help='Square side length. Must be a multiple of 16.')
    # Training
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=8)
    # I/O
    p.add_argument('--output_dir', type=str, required=True,
                   help='Where to save the checkpoint and hparams.json files')
    p.add_argument('--resume', action='store_true',
                   help='If checkpoint exists in output_dir, resume from it')
    # Logging
    p.add_argument('--wandb_project', type=str, default=None,
                   help='W&B project name. If none, W&B is disabled.')
    p.add_argument('--wandb_run_name', type=str, default=None,
                   help='Optional run name, defaults to W&B auto-generated.')
    return p.parse_args()

def set_seed(seed):
    """Set seeds for reproducibility. Not fully deterministic on GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_dataloaders(args):
    """Train + val DataLoaders.
    Transforms: resize so the short side is img_size, center crop to square,
    convert to [0, 1] tensor. Matches the [0, 1] range that the get_visual
    function in Shell produces.
    """
    transform = Compose([
        Resize(args.img_size),
        CenterCrop(args.img_size),
        ToTensor(), # uint8 HWC -> float CHW in [0, 1]
    ])

    train_set = ImageDirDataset(args.train_dir, transform=transform)
    val_set = ImageDirDataset(args.val_dir, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                            # drop last partial batch
    )

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f'Train: {len(train_set)} images | Val: {len(val_set)} images')
    return train_loader, val_loader

def save_hparams(args, output_dir):
    """Write the hyperparameters used as a JSON sidecar next to the checkpoint."""
    hparams = vars(args).copy()
    with open(output_dir/'hparams.json', 'w') as f:
        json.dump(hparams, f, indent=2)

def train_step(model, x, optimizer):
    """One optimizer step. Returns loss value."""
    x_hat, _ = model(x)
    loss = F.mse_loss(x_hat, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad() # wrap the function in no_grad
def eval_epoch(model, val_loader, device):
    """Mean val loss over the held-out set (per-pixel)."""
    model.eval()
    total, n = 0.0, 0
    for x, _ in val_loader: # iterate over images, discard labels
        x = x.to(device, non_blocking=True) # copy batch from CPU to GPU, CPU
                                        # does not wait to execute next line
        x_hat, _ = model(x) # forward pass, return reconstruction, discard latent
        loss = F.mse_loss(x_hat, x, reduction='sum') # compute MSE loss, summing
                                        # squared errors across all dims
        total += loss.item() # sum of squared errors across validation set
        n += x.numel() # total number of pixel-channel values across all validation
    model.train() # switch back to train mode for next epoch
    return total / n # per-pixel mean

def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir/'checkpoint.pt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    train_loader, val_loader = build_dataloaders(args)
    model = Res18AE(latent_dim=args.latent_dim, img_size=args.img_size).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    start_epoch = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from {checkpoint_path} at epoch {start_epoch}.')
    else:
        save_hparams(args, output_dir)

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            config=vars(args), resume='allow',
        )

    model.train()
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch {epoch}')
        for step, (x, _) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            loss = train_step(model, x, optimizer)

            if step % 50 == 0:
                print(f'  step {step:>5d} | train_loss {loss:.4f}')
                if use_wandb:
                    wandb.log({'train_loss': loss, 'epoch': epoch, 'step': step})

        val_loss = eval_epoch(model, val_loader, device)
        print(f'Epoch {epoch} | val_loss {val_loss:.4f}')
        if use_wandb:
            wandb.log({'val_loss': val_loss, 'epoch': epoch})

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, checkpoint_path)
        print(f'Saved {checkpoint_path}')
    if use_wandb:
        wandb.finish()
    

if __name__ == '__main__':
    main()
