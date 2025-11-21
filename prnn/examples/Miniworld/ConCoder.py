import math
import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

from typing import Tuple, Sequence, Optional

from prnn.utils.thetaRNN import LayerNormRNNCell


class ConCoder(nn.Module):
    """Contrastive encoder.

    Expected input to `forward` is a batch of image sequences with shape
        (B, S, C, H, W)
    where S >= k + T (k = context steps, T = prediction steps).

    This module provides:
    - an image encoder (CNN) that maps each frame to a latent vector (D)
    - a vanilla RNN (nn.RNN) that consumes k latents and predicts T future latents
    - a linear prediction head mapping RNN hidden -> latent space
    """

    def __init__(
        self,
        learning_rate: float,
        net_config: Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int]],
        in_channels: int,
        latent_dim: int,
        context_steps: int = 4,
        pred_steps: int = 3,
        hidden_size: Optional[int] = None,
        encoder_weight_decay: float = 1e-5,
        **cell_kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.context_steps = context_steps
        self.pred_steps = pred_steps
        self.encoder_weight_decay = encoder_weight_decay

        n_channels, kernel_sizes, strides, paddings, output_paddings = net_config

        # build encoder (similar to VAE encoder but outputs deterministic latent)
        self._build_encoder(in_channels, n_channels, kernel_sizes, strides, paddings)

        # RNN predictor
        rnn_hidden = hidden_size or latent_dim
        # NOTE: may include other RNN options later
        self.rnn = LayerNormRNNCell(input_size=latent_dim, hidden_size=rnn_hidden, **cell_kwargs)
        self.pred_head = nn.Linear(rnn_hidden, latent_dim)

        self.initialize_weights()

        # optimizer: apply weight decay only to encoder params (including fc)
        # NOTE: maybe all parameters should be decayed?
        encoder_params = list(self.encoder.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]
        self.optimizer = torch.optim.Adam(
            [
                {"params": encoder_params, "weight_decay": self.encoder_weight_decay},
                {"params": other_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )

    def _build_encoder(self, in_channels, n_channels, kernel_sizes, strides, paddings):
        modules = []

        # CNN layers
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

        # Flatten and project to latent space
        modules.append(nn.Flatten())
        modules.append(nn.Linear(n_channels[-1] * 16 * 16, self.latent_dim))
        modules.append(self.activation)
        self.encoder = nn.Sequential(*modules)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

        # RNN and pred_head init (RNN is initialized in LayerNormRNNCell
        # for name, p in self.rnn.named_parameters():
        #     if 'weight' in name:
        #         nn.init.orthogonal_(p)
        #     elif 'bias' in name:
        #         nn.init.zeros_(p)
        nn.init.xavier_normal_(self.pred_head.weight)
        if self.pred_head.bias is not None:
            nn.init.zeros_(self.pred_head.bias)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latents.

        frames: (B, S, C, H, W) -> returns latents (B, S, D)
        """
        B, S, C, H, W = frames.shape
        x = frames.view(B * S, C, H, W)
        lat = self.encoder(x)
        lat = lat.view(B, S, -1)
        return lat

    def predict_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Given latents (B, S, D), use first k=self.context_steps to predict next T=self.pred_steps.

        Returns preds of shape (B, T, D) and the corresponding ground-truth targets of shape (B, T, D).
        Requires S >= k + T.
        """
        B, S, D = latents.shape
        k = self.context_steps
        T = self.pred_steps
        assert S >= k + T, "Need sequence length S >= context + pred steps"

        # context -> rnn
        context = latents[:, :k, :]  # B, k, D
        _, hidden = self.rnn(context)  # hidden: (num_layers, B, hidden)
        # take last layer hidden
        h = hidden[-1]  # (B, hidden)

        preds = []
        hx = h
        for _ in range(T):
            p = self.pred_head(hx)  # (B, D)
            preds.append(p)
            # feed back predicted latent to rnn for next step
            out, hidden = self.rnn(p.unsqueeze(1), hidden)
            hx = hidden[-1]

        preds = torch.stack(preds, dim=1)  # B, T, D
        targets = latents[:, k:k + T, :]
        return preds, targets


def contrastive_loss_from_preds(preds: torch.Tensor, targets: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute a simple contrastive loss: for each predicted vector, the matching target at same (B,t) is positive;
    all other target vectors are negatives. Uses dot-product similarity and cross-entropy.

    preds: (B, T, D)
    targets: (B, T, D)
    """
    B, T, D = preds.shape
    preds_flat = preds.reshape(B * T, D)  # (N, D)
    targets_flat = targets.reshape(B * T, D)  # (N, D)

    # similarity matrix (N, N)
    logits = torch.matmul(preds_flat, targets_flat.T) / temperature

    labels = torch.arange(B * T, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss


class ConCoderLightning(pl.LightningModule):
    """Lightning wrapper for ConCoder for faster pretraining.

    Expects training batches of shape (B, S, C, H, W). The training_step computes
    the contrastive loss described above using the module's predict_latents.
    """

    def __init__(
        self,
        learning_rate: float,
        net_config,
        in_channels: int,
        latent_dim: int,
        context_steps: int = 4,
        pred_steps: int = 3,
        encoder_weight_decay: float = 1e-5,
    ):
        super().__init__()
        self._learning_rate = learning_rate
        self.context_steps = context_steps
        self.pred_steps = pred_steps
        self.encoder_weight_decay = encoder_weight_decay

        # create encoder/predictor
        self.model = ConCoder(
            learning_rate=learning_rate,
            net_config=net_config,
            in_channels=in_channels,
            latent_dim=latent_dim,
            context_steps=context_steps,
            pred_steps=pred_steps,
            encoder_weight_decay=encoder_weight_decay,
        )

        # save hyperparams except big objects
        self.save_hyperparameters(ignore=["net_config"])

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.model.encode_frames(frames)

    def training_step(self, batch, batch_idx):
        # Expect batch to be (frames, labels) or just frames
        if isinstance(batch, (list, tuple)):
            frames = batch[0]
        else:
            frames = batch

        # frames: B, S, C, H, W
        lat = self.model.encode_frames(frames)
        preds, targets = self.model.predict_latents(lat)
        loss = contrastive_loss_from_preds(preds, targets)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # apply weight decay only to encoder parameters (include fc_latent)
        encoder_params = list(self.model.encoder.parameters()) + [self.model.fc_latent.weight, self.model.fc_latent.bias]
        encoder_param_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]
        return torch.optim.Adam(
            [
                {"params": encoder_params, "weight_decay": self.encoder_weight_decay},
                {"params": other_params, "weight_decay": 0.0},
            ],
            lr=self._learning_rate,
        )
