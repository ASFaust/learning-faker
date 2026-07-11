"""Loss.

    pinball = quantile loss of y vs the predicted absolute quantiles, at each tau

The head models the full predictive distribution directly as absolute quantiles
of the (per-task-normalized) log-loss. Pinball is bounded-influence (L1-like) so
diverged runs don't destabilize it, its tau=0.5 term is exactly the median (the
robust point estimate), and averaged over a tau-grid it approximates CRPS / W1 --
a proper, non-NLL, Wasserstein-flavored objective. There is no separate mean head:
median-centric by design, so the point estimate is always inside its own band.
"""

from __future__ import annotations

import torch


def pinball_loss(pred, y: torch.Tensor, taus: torch.Tensor):
    """Quantile loss of y against the predicted absolute quantiles.

    pred.quantiles: (B, 2, Q) absolute quantiles. y: (B, 2). taus: (Q,).
    Returns (loss, metrics).
    """
    diff = y.unsqueeze(-1) - pred.quantiles          # (B, 2, Q)
    t = taus.view(1, 1, -1)
    pin = torch.maximum(t * diff, (t - 1) * diff)    # (B, 2, Q)
    loss = pin.mean()
    return loss, {"pinball": loss.detach()}
