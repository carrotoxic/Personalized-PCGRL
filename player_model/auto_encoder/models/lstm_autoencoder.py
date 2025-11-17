from typing import Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig


class LSTMAutoencoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder_lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
        )


        self.to_latent = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )


        self.decoder_lstm = nn.LSTM(
            input_size=cfg.latent_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, cfg.input_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for LSTM
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder_lstm(x)
        device = outputs.device
        lengths = lengths.to(device)

        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(2))
        last_outputs = outputs.gather(1, idx).squeeze(1)

        latent = self.to_latent(last_outputs)
        return latent

    def decode(self, latent: torch.Tensor, max_len: int) -> torch.Tensor:
        B = latent.size(0)
        device = latent.device

        z_rep = latent.unsqueeze(1).expand(B, max_len, -1)
        dec_outputs, _ = self.decoder_lstm(z_rep)

        recon = self.output_proj(dec_outputs)
        return recon

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x, lengths)
        recon = self.decode(latent, x.size(1))
        return recon, latent