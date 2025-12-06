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
        n_targets: int = 32,
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
        self.n_targets = n_targets
        self.encoder_weight_decay = encoder_weight_decay

        n_channels, kernel_sizes, strides, paddings, output_paddings = net_config

        # build encoder (similar to VAE encoder but outputs deterministic latent)
        self._build_encoder(in_channels, n_channels, kernel_sizes, strides, paddings)

        # RNN predictor
        rnn_hidden = hidden_size or latent_dim
        # NOTE: may include other RNN options later
        self.rnn_cell = LayerNormRNNCell(input_size=latent_dim, hidden_size=rnn_hidden, **cell_kwargs)
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
        
    def rnn(self, x, hx=None):
        # timestep dimension first
        x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()
        if hx is None:
            hx = torch.zeros(1, batch_size, self.rnn_cell.hidden_size,
                             device=x.device) # 1, B*N, R for the sake of LayerNormRNNCell
        h_t_minus_1 = hx.clone()
        output = []
        for t in range(seq_len):
            h_t, _ = self.rnn_cell(x[t], torch.zeros_like(h_t_minus_1,
                                                          device=h_t_minus_1.device),
                                   h_t_minus_1)
            output.append(h_t.clone())
            h_t_minus_1 = h_t.clone()
        output = torch.stack(output)

        output = output.transpose(0, 1)
        return output, h_t

    def predict_latents(self, latents: torch.Tensor):
        """Given latents (B, S, D), use first k=self.context_steps to predict next T=self.pred_steps.

        Returns preds of shape (B, T, D) and the corresponding ground-truth targets of shape (B, T, D).
        Requires S >= k + T.
        """
        B, S, D = latents.shape
        k = self.context_steps
        T = self.pred_steps
        Z = self.n_targets
        assert S >= k + T, "Need sequence length S >= context + pred steps"
        assert S >= Z, "Need sequence length S >= n_targets"
        N = (S - T) // k

        preds = torch.zeros((B, N, T, D), device=latents.device)
        targets = torch.zeros((B, N, Z, D), device=latents.device)

        # context -> rnn
        context = latents[:, :k*N, :]  # B, k*N, D
        context = context.reshape(B*N, k, D)
        _, hidden = self.rnn(context)  # hidden: (B*N, hidden)

        for i in range(T):
            p = self.pred_head(hidden)  # (B*N, D)
            preds[:, :, i, :] = p.view(B, N, D)
            # feed back predicted latent to rnn for next step
            out, hidden = self.rnn(p.transpose(0,1), hidden)

        for i in range(N):
            idx = (i+1)*k + 1
            targets[:, i, :T, :]  = latents[:, idx:idx+T, :]  # B, T, D
            if idx + 1 < Z//2:
                targets[:, i, T:T+idx, :]  = latents[:, :idx, :]  # B, Z-x, D
                targets[:, i, T+idx:, :]  = latents[:, idx+T:Z, :]  # B, x-T, D
            elif S - (idx+T//2) < Z//2:
                x = Z - (S - idx + T)
                targets[:, i, T:(S-idx), :]  = latents[:, idx+T:, :]  # B, Z-x-T, D
                targets[:, i, T+(S-idx):, :]  = latents[:, S - x:S, :]  # B, x, D
            else:
                targets[:, i, T:(Z+T)//2, :]  = latents[:, idx-(Z-T)//2:idx, :]  # B, (Z-T)//2, D
                targets[:, i, (Z+T)//2:, :]  = latents[:, idx+T:idx+Z-(Z-T)//2, :]  # B, ~(Z-T)//2, D

        preds = preds.view(B*N, T, D)
        targets = targets.view(B*N, Z, D)
                
        return preds, targets


def contrastive_loss_from_preds(preds: torch.Tensor, targets: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute a simple contrastive loss: for each predicted vector, the matching target at same (B,t) is positive;
    all other target vectors are negatives. Uses dot-product similarity and cross-entropy.

    preds: (B*N, T, D)
    targets: (B*N, Z, D)
    """
    B, T, D = preds.shape
    
    loss = []
    labels = torch.arange(T, device=preds.device)
    
    for b in range(B):
        # similarity matrix (T, Z)
        logits = torch.matmul(preds[b], targets[b].T) / temperature
        loss.append(F.cross_entropy(logits, labels))
    loss = torch.stack(loss).mean()
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
        n_targets: int = 32,
        hidden_size=None,
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
            n_targets=n_targets,
            hidden_size=hidden_size,
            encoder_weight_decay=encoder_weight_decay,
        )

        # save hyperparams except big objects
        self.save_hyperparameters(ignore=["net_config"])

    def forward(self, frames: torch.Tensor):
        # frames: B, S, C, H, W
        lat = self.model.encode(frames)
        preds, targets = self.model.predict_latents(lat)
        return preds, targets

    def training_step(self, frames):
        # frames: B, S, C, H, W
        preds, targets = self.forward(frames)
        loss = contrastive_loss_from_preds(preds, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # apply weight decay only to encoder parameters
        encoder_params = list(self.model.encoder.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.parameters() if id(p) not in encoder_param_ids]
        return torch.optim.Adam(
            [
                {"params": encoder_params, "weight_decay": self.encoder_weight_decay},
                {"params": other_params, "weight_decay": 0.0},
            ],
            lr=self._learning_rate,
        )
