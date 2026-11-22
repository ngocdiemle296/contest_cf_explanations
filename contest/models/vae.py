import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def resource_feasibility_loss(x_hat, mask, compat_matrix):
    from contest.data import event_log as el
    n_act = len(el.ACTIVITIES)
    n_res = len(el.RESOURCES)
    act_probs = torch.softmax(x_hat[..., :n_act], dim=-1)
    res_probs = torch.softmax(x_hat[..., n_act:n_act + n_res], dim=-1)
    M = compat_matrix.to(x_hat.device)
    feasibility = torch.einsum('btr,ra,bta->bt', res_probs, M, act_probs)
    if mask is not None:
        feasibility = feasibility * mask
        score = feasibility.sum() / mask.sum().clamp(min=1)
    else:
        score = feasibility.mean()
    return -score

def diff_ero_loss(x_hat, act_seq, mask, PM_ground_truth):
    n_act = PM_ground_truth.size(0)
    O = F.softmax(x_hat[..., :n_act], dim=-1)

    flat_idx  = act_seq[:, :-1].reshape(-1)
    flat_O    = O[:, 1:, :].reshape(-1, n_act)
    flat_mask = (mask[:, :-1] * mask[:, 1:]).reshape(-1)

    O_b = torch.zeros(n_act, n_act, device=O.device)
    counts = torch.zeros(n_act, device=O.device)
    O_b.index_add_(0, flat_idx, flat_O * flat_mask.unsqueeze(-1))
    counts.index_add_(0, flat_idx, flat_mask)

    visited = counts > 0
    O_b[visited] = O_b[visited] / counts[visited].unsqueeze(-1)

    ce = -(PM_ground_truth[visited] * torch.log(O_b[visited].clamp(min=1e-8))).sum(dim=-1).mean()
    return ce

class LSTMEncoder(nn.Module):

    def __init__(self, event_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=event_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc_mu    = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)

        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class LSTMDecoder(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int, event_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.z_to_h0 = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.z_to_c0 = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.lstm = nn.LSTM(
            input_size=event_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, event_dim)

    def forward(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h0 = self.z_to_h0(z).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = self.z_to_c0(z).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        out, _ = self.lstm(target, (h0, c0))
        return self.output_proj(out)

    def decode_from_z(self, z: torch.Tensor, seq_len: int, event_dim: int, device: torch.device) -> torch.Tensor:
        B = z.size(0)
        h0 = self.z_to_h0(z).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = self.z_to_c0(z).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        inp = torch.zeros(B, 1, event_dim, device=device)
        outputs = []
        h, c = h0, c0
        for _ in range(seq_len):
            out, (h, c) = self.lstm(inp, (h, c))
            step = self.output_proj(out)
            outputs.append(step)
            inp = step.detach()
        return torch.cat(outputs, dim=1)


class LSTMVAE(nn.Module):

    def __init__(
        self,
        event_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        beta: float = 1.0,
        gamma: float = 0.0,
        gamma_conf: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma

        self.register_buffer("PM_ground_truth", None)
        self.register_buffer("gamma_conf_tensor", torch.tensor(float(gamma_conf)))
        self._gamma_conf_cached: float = float(gamma_conf)

        from contest.data import event_log as el
        if el.RESOURCE_ACTIVITY_MATRIX.size > 0:
            self.register_buffer(
                "compat_matrix",
                torch.tensor(el.RESOURCE_ACTIVITY_MATRIX, dtype=torch.float32),
            )
        else:
            self.compat_matrix = None

        self.encoder = LSTMEncoder(event_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, event_dim, num_layers)

    def set_ground_truth_transitions(self, PM: torch.Tensor, gamma_conf: float = 0.5) -> None:
        self.PM_ground_truth = PM
        self.gamma_conf_tensor.fill_(gamma_conf)
        self._gamma_conf_cached = float(gamma_conf)

    @property
    def gamma_conf(self) -> float:
        return self._gamma_conf_cached


    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, lengths)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode_trace(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x, lengths)
        return mu


    def decode(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, target)

    def decode_autoregressive(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        return self.decoder.decode_from_z(z, seq_len, self.decoder.output_proj.out_features, z.device)

    def build_decoder_input(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        epoch: int,
        total_epochs: int,
        warmup_frac: float = 0.15,
        k: float = 2.0,
        force_tf_prob: Optional[float] = None,
    ) -> torch.Tensor:
        if force_tf_prob is not None:
            teacher_forcing_prob = force_tf_prob
        else:
            warmup_epochs = int(warmup_frac * total_epochs)
            eff_epoch = max(0, epoch - warmup_epochs)
            eff_total = max(1, total_epochs - warmup_epochs)

            if epoch < warmup_epochs:
                teacher_forcing_prob = 1.0
            else:
                teacher_forcing_prob = k / (k + torch.exp(torch.tensor(eff_epoch / eff_total * k)))
                teacher_forcing_prob = teacher_forcing_prob.item()

        shifted_gt = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)

        if teacher_forcing_prob >= 0.999:
            return shifted_gt, teacher_forcing_prob

        with torch.no_grad():
            x_hat_peek = self.decode(z, shifted_gt)
        shifted_pred = torch.cat(
            [torch.zeros_like(x_hat_peek[:, :1, :]), x_hat_peek[:, :-1, :]], dim=1
        ).detach()

        use_teacher = (
            torch.rand(x.size(0), x.size(1), 1, device=x.device) < teacher_forcing_prob
        )
        return torch.where(use_teacher, shifted_gt, shifted_pred), teacher_forcing_prob


    def elbo_loss(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        gamma_conf: Optional[float] = None,
        epoch: int = 0,
        total_epochs: int = 1,
        use_scheduled_sampling: bool = True,
        force_tf_prob: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, lengths)
        z = self.reparameterise(mu, logvar)

        if force_tf_prob is not None:
            dec_input, teacher_forcing_prob = self.build_decoder_input(
                x, z, epoch=epoch, total_epochs=total_epochs,force_tf_prob=force_tf_prob
            )
        elif use_scheduled_sampling:
            dec_input, teacher_forcing_prob = self.build_decoder_input(x, z, epoch=epoch, total_epochs=total_epochs)
        else:
            dec_input = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
            teacher_forcing_prob = 1.0
        x_hat = self.decode(z, dec_input)

        recon = F.mse_loss(x_hat * mask.unsqueeze(-1), x * mask.unsqueeze(-1), reduction='sum')
        recon = recon / mask.sum()

        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        if self._gamma_conf_cached > 0.0 and self.PM_ground_truth is not None:
            act_seq = x[..., :self.PM_ground_truth.size(0)].argmax(dim=-1)
            ero_loss = diff_ero_loss(x_hat, act_seq, mask, self.PM_ground_truth.to(x_hat.device))

            if self.compat_matrix is not None:
                res_loss = resource_feasibility_loss(x_hat, mask, self.compat_matrix)
                conf_loss = ero_loss + res_loss
            else:
                conf_loss = ero_loss
        else:
            conf_loss = torch.tensor(0.0, device=x.device)

        loss = recon + self.beta * kl + self._gamma_conf_cached * conf_loss
        return loss, recon, kl, conf_loss, teacher_forcing_prob

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor):
        return self.elbo_loss(x, lengths, mask)
