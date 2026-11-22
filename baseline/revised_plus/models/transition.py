import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TransitionModel(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        n_activities: int,
        n_resources: int,
        act_embed_dim: int = 8,
        res_embed_dim: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.act_embed = nn.Embedding(n_activities, act_embed_dim)
        self.res_embed = nn.Embedding(n_resources, res_embed_dim)

        in_dim = latent_dim + act_embed_dim + res_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.log_noise = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        z: torch.Tensor,
        act_idx: torch.Tensor,
        res_idx: torch.Tensor,
    ) -> torch.Tensor:
        a_emb = self.act_embed(act_idx)
        r_emb = self.res_embed(res_idx)
        inp = torch.cat([z, a_emb, r_emb], dim=-1)
        delta = self.net(inp)
        return z + delta

    def noise_std(self) -> torch.Tensor:
        return torch.exp(self.log_noise)

    def transition_loss(
        self,
        z_t: torch.Tensor,
        act_idx: torch.Tensor,
        res_idx: torch.Tensor,
        z_t1: torch.Tensor,
    ) -> torch.Tensor:
        z_pred = self.forward(z_t, act_idx, res_idx)
        mse = F.mse_loss(z_pred, z_t1)
        return mse

    @torch.no_grad()
    def simulate(
        self,
        z0: torch.Tensor,
        act_seq: torch.Tensor,
        res_seq: torch.Tensor,
        stochastic: bool = False,
    ) -> torch.Tensor:
        B = z0.size(0)
        K = act_seq.size(-1)
        if act_seq.dim() == 1:
            act_seq = act_seq.unsqueeze(0).expand(B, -1)
            res_seq = res_seq.unsqueeze(0).expand(B, -1)

        traj = [z0]
        z = z0
        for k in range(K):
            z_next = self.forward(z, act_seq[:, k], res_seq[:, k])
            if stochastic:
                z_next = z_next + self.noise_std() * torch.randn_like(z_next)
            traj.append(z_next)
            z = z_next
        return torch.stack(traj, dim=1)


class RewardFunction(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        n_activities: int,
        n_resources: int,
        act_embed_dim: int = 8,
        res_embed_dim: int = 4,
        hidden_dim: int = 64,
        w_outcome: float = 1.0,
        w_effort: float = 0.3,
        w_compliance: float = 0.5,
    ):
        super().__init__()
        self.w_outcome    = w_outcome
        self.w_effort     = w_effort
        self.w_compliance = w_compliance

        self.act_embed = nn.Embedding(n_activities, act_embed_dim)
        self.res_embed = nn.Embedding(n_resources, res_embed_dim)

        in_dim = latent_dim + act_embed_dim + res_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        act_idx: torch.Tensor,
        res_idx: torch.Tensor,
    ) -> torch.Tensor:
        a_emb = self.act_embed(act_idx)
        r_emb = self.res_embed(res_idx)
        inp = torch.cat([z, a_emb, r_emb], dim=-1)
        return self.net(inp).squeeze(-1)

    @torch.no_grad()
    def expected_return(
        self,
        z_traj: torch.Tensor,
        act_seq: torch.Tensor,
        res_seq: torch.Tensor,
        gamma: float = 0.95,
    ) -> torch.Tensor:
        K_actual = act_seq.size(1)
        total_return = torch.tensor(0.0, device=z_traj.device)

        for k in range(K_actual):
            z_k = z_traj[:, k, :]
            a_k = act_seq[:, k]
            r_k = res_seq[:, k]

            r_step = self.forward(z_k, a_k, r_k).item()
            total_return += (gamma ** k) * r_step

        return total_return
