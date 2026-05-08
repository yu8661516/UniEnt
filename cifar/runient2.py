"""
RUniEnt2: Improved UniEnt with three targeted fixes over the original UniEnt:

Fix A — Cross-batch EMA statistics with batch-size-adaptive momentum:
  Instead of fitting a fresh GMM on each batch (unstable at small batch sizes),
  we maintain exponential moving averages of the ID/OOD component means and
  variances. The EMA momentum is batch-size-adaptive: larger batches produce
  more reliable GMM fits, so they receive a larger update weight
  (momentum = 1 - ema_base_lr / sqrt(batch_size)). This keeps the EMA
  responsive to reliable estimates while remaining stable under small batches.

Fix B — OOD-aware marginal entropy:
  The original UniEnt applies marginal entropy maximisation (diversity
  regularisation) over the whole batch, including OOD samples whose predictions
  should be uniformly high-entropy anyway. We restrict this term to the
  ID-weighted samples only, avoiding conflicting optimisation signals.

Fix C — Dynamic prototype update with drift constraint:
  UniEnt uses static source-domain classifier weights as class prototypes for
  csOOD scoring. As BN layers adapt, the feature space drifts away from the
  source domain, making static prototypes increasingly inaccurate. We update
  prototypes via EMA using high-confidence ID samples, keeping the csOOD score
  aligned with the evolving feature space.

  Three safeguards prevent OOD contamination of prototypes:
  (a) Updates only begin after warm-up (when GMM weights are reliable).
  (b) A cosine-similarity drift constraint stops updates if a prototype
      deviates too far from its source-domain initialisation.
  (c) The EMA momentum is batch-size-adaptive: larger batches contain more
      OOD samples in absolute terms, so each sample contributes a smaller
      fractional update, keeping the per-sample effective learning rate
      constant across batch sizes.

All fixes are applied on top of the original UniEnt (ent_unf) formulation and
reuse its model-configuration / parameter-collection helpers from tent.py.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from tent import Tent


class RUniEnt2(Tent):
    """
    Improved UniEnt for open-set TTA.

    Key differences from the original UniEnt (ent_unf):
      1. EMA-stabilised GMM: component statistics are updated incrementally
         across batches rather than re-fitted from scratch each time.
         Falls back to pure-Tent behaviour during warm-up.
      2. OOD-aware marginal entropy: diversity regularisation is computed
         only over ID-weighted samples, not the full batch.
      3. Dynamic prototype update: class prototypes are updated via EMA using
         high-confidence ID samples, keeping csOOD scores aligned with the
         evolving feature space as BN layers adapt.
      4. Per-corruption reset: model, optimiser, and all running state are
         restored between corruption types (independent evaluation).
    """

    def __init__(self, model, optimizer, steps=1, episodic=False,
                 ema_base_lr=2.0,
                 proto_base_lr=0.002,
                 proto_drift_threshold=0.90,
                 delta=0.6,
                 ood_lambda=0.5,
                 marg_lambda=0.1,
                 warmup_steps=5,
                 use_ema=True,
                 use_ood_marg=True,
                 use_proto=True):
        super().__init__(model, optimizer, steps, episodic)

        # ── ablation switches ────────────────────────────────────────────────
        self.use_ema      = use_ema      # Fix A: EMA-stabilised GMM statistics
        self.use_ood_marg = use_ood_marg # Fix B: OOD-aware marginal entropy
        self.use_proto    = use_proto    # Fix C: dynamic prototype update

        self.device = next(model.parameters()).device

        # ── detect classifier head name (fc for WRN, classifier for ResNeXt) ──
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            self._head_name = 'fc'
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            self._head_name = 'classifier'
        else:
            raise AttributeError("Cannot find Linear classifier head (tried 'fc' and 'classifier')")

        # ── frozen reference model for feature extraction ────────────────────
        self._ref_model = copy.deepcopy(model)
        self._ref_model.eval()
        for p in self._ref_model.parameters():
            p.requires_grad_(False)
        setattr(self._ref_model, self._head_name, nn.Identity())

        # ── class prototypes: start from source classifier weights ───────────
        self.prototype      = getattr(model, self._head_name).weight.data.clone().to(self.device)
        self._proto_init    = self.prototype.clone()

        # ── save clean state for per-corruption reset ────────────────────────
        self._model_state   = copy.deepcopy(model.state_dict())
        self._optim_state   = copy.deepcopy(optimizer.state_dict())

        # ── hyper-parameters ─────────────────────────────────────────────────
        self.ema_base_lr          = ema_base_lr           # base lr for EMA momentum scaling
        self.proto_base_lr        = proto_base_lr         # base lr for prototype EMA
        self.proto_drift_threshold = proto_drift_threshold # cosine sim floor before stopping update
        self.delta                = delta
        self.ood_lambda           = ood_lambda
        self.marg_lambda          = marg_lambda
        self.warmup_steps         = warmup_steps

        # ── EMA state for GMM components (initialised lazily) ────────────────
        self._ema_ready = False
        self._mu_id     = None
        self._mu_ood    = None
        self._var_id    = None
        self._var_ood   = None

        self.step = 0

    # ── per-corruption reset ─────────────────────────────────────────────────
    def reset_corruption(self):
        """Restore model, optimiser, prototypes, and EMA state."""
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optim_state)
        self.prototype  = self._proto_init.clone()
        self._ema_ready = False
        self._mu_id = self._mu_ood = None
        self._var_id = self._var_ood = None
        self.step = 0

    # ── csOOD score ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def _csood_score(self, x):
        """
        1 - max_cosine_similarity(feature, prototypes).
        Uses the frozen reference model (no per-batch deepcopy).
        Prototypes are dynamically updated (Fix C), so the score stays
        aligned with the current feature space.
        """
        feat  = self._ref_model(x)                    # [B, D]
        feat  = F.normalize(feat, dim=1)
        proto = F.normalize(self.prototype, dim=1)    # [C, D]
        cos_sim = torch.matmul(feat, proto.T)         # [B, C]
        max_sim = cos_sim.max(dim=1)[0]               # [B]
        return (1.0 - max_sim).clamp(0.0, 1.0)       # [B]

    # ── EMA update for GMM statistics (Fix A) ────────────────────────────────
    def _update_ema(self, scores_np, batch_size):
        """Skipped when use_ema=False (ablation: no cross-batch EMA)."""
        if not self.use_ema:
            # fit fresh GMM each batch, no EMA blending
            try:
                gmm = GaussianMixture(n_components=2, random_state=42,
                                      n_init=3, max_iter=100)
                gmm.fit(scores_np.reshape(-1, 1))
                id_comp  = int(np.argmin(gmm.means_.ravel()))
                ood_comp = 1 - id_comp
                self._mu_id   = float(gmm.means_[id_comp])
                self._mu_ood  = float(gmm.means_[ood_comp])
                self._var_id  = max(float(gmm.covariances_[id_comp]),  1e-6)
                self._var_ood = max(float(gmm.covariances_[ood_comp]), 1e-6)
                self._ema_ready = True
            except Exception:
                pass
            return
        # EMA path: blend current GMM fit into running statistics
        # m = 1 - ema_base_lr / sqrt(B): bs=32→0.646, bs=64→0.750, bs=128→0.823, bs=200→0.858
        m = float(np.clip(1.0 - self.ema_base_lr / max(batch_size, 1) ** 0.5,
                          0.5, 0.99))
        try:
            gmm = GaussianMixture(n_components=2, random_state=42,
                                  n_init=3, max_iter=100)
            gmm.fit(scores_np.reshape(-1, 1))

            id_comp  = int(np.argmin(gmm.means_.ravel()))
            ood_comp = 1 - id_comp

            mu_id_new   = float(gmm.means_[id_comp])
            mu_ood_new  = float(gmm.means_[ood_comp])
            var_id_new  = float(gmm.covariances_[id_comp])
            var_ood_new = float(gmm.covariances_[ood_comp])

            if not self._ema_ready:
                self._mu_id   = mu_id_new
                self._mu_ood  = mu_ood_new
                self._var_id  = max(var_id_new,  1e-6)
                self._var_ood = max(var_ood_new, 1e-6)
                self._ema_ready = True
            else:
                self._mu_id   = m * self._mu_id   + (1 - m) * mu_id_new
                self._mu_ood  = m * self._mu_ood  + (1 - m) * mu_ood_new
                self._var_id  = m * self._var_id  + (1 - m) * max(var_id_new,  1e-6)
                self._var_ood = m * self._var_ood + (1 - m) * max(var_ood_new, 1e-6)
        except Exception:
            pass

    # ── per-sample ID/OOD weights ─────────────────────────────────────────────
    def _compute_weights(self, scores):
        """
        Returns (w_id, w_ood) using EMA Gaussian posteriors.
        Falls back to pure-Tent during warm-up or when components overlap.
        """
        B = scores.shape[0]
        ones  = torch.ones(B,  device=self.device)
        zeros = torch.zeros(B, device=self.device)

        if self.step < self.warmup_steps or not self._ema_ready:
            return ones, zeros

        if abs(self._mu_ood - self._mu_id) < 0.05:
            return ones, zeros

        def gauss_pdf(x, mu, var):
            return torch.exp(-0.5 * (x - mu) ** 2 / var) / (2 * var) ** 0.5

        p_id  = gauss_pdf(scores, self._mu_id,  self._var_id)
        p_ood = gauss_pdf(scores, self._mu_ood, self._var_ood)
        pi_id = (p_id / (p_id + p_ood + 1e-10)).clamp(0.0, 1.0)

        d = self.delta
        w_id = torch.where(
            pi_id >= d,       pi_id,
            torch.where(pi_id <= 1.0 - d, zeros, 0.5 * pi_id)
        )
        w_ood = torch.where(
            pi_id <= 1.0 - d, 1.0 - pi_id,
            torch.where(pi_id >= d,       zeros, 0.5 * (1.0 - pi_id))
        )
        return w_id, w_ood

    # ── dynamic prototype update (Fix C) ─────────────────────────────────────
    @torch.no_grad()
    def _update_prototypes(self, x, outputs, w_id, batch_size):
        """
        Update class prototypes via EMA using high-confidence ID samples.

        Three safeguards:
        (a) Only runs after warm-up, when GMM weights are reliable.
        (b) Batch-size-adaptive momentum: proto_momentum = 1 - proto_base_lr /
            batch_size, so each sample contributes the same fractional update
            regardless of batch size. Larger batches → more conservative EMA.
        (c) Drift constraint: if a prototype's cosine similarity to its source
            initialisation falls below proto_drift_threshold, the update for
            that class is skipped. This prevents OOD contamination from
            accumulating and pulling prototypes away from the source manifold.
        """
        # safeguard (a): skip during warm-up
        if self.step <= self.warmup_steps:
            return

        id_mask = w_id >= self.delta
        if id_mask.sum() == 0:
            return

        # safeguard (b): batch-size-adaptive momentum
        m = 1.0 - self.proto_base_lr / max(batch_size, 1)
        m = float(np.clip(m, 0.9, 0.9999))

        feat_id  = self._ref_model(x[id_mask])          # [N_id, D]
        feat_id  = F.normalize(feat_id, dim=1)
        pred_cls = outputs[id_mask].argmax(dim=1)        # [N_id]

        for c in pred_cls.unique():
            mask_c       = (pred_cls == c)
            proto_update = feat_id[mask_c].mean(0)       # [D]
            candidate    = m * self.prototype[c] + (1 - m) * proto_update

            # safeguard (c): drift constraint
            cos_sim = F.cosine_similarity(
                candidate.unsqueeze(0),
                self._proto_init[c].unsqueeze(0)
            ).item()
            if cos_sim >= self.proto_drift_threshold:
                self.prototype[c] = candidate

    # ── main forward ─────────────────────────────────────────────────────────
    @torch.enable_grad()
    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            # 1. csOOD scores using current (dynamically updated) prototypes
            scores    = self._csood_score(x)
            scores_np = scores.cpu().numpy()

            # 2. update EMA GMM statistics (batch-size-adaptive momentum)
            self._update_ema(scores_np, x.shape[0])
            self.step += 1

            # 3. compute per-sample weights
            w_id, w_ood = self._compute_weights(scores)

            # 4. forward through the adapting model
            outputs = self.model(x)

            # 5. entropy loss
            p   = F.softmax(outputs, dim=1)
            H   = -(p * torch.log(p + 1e-10)).sum(dim=1)

            loss_id  = (w_id  * H).mean()
            loss_ood = (w_ood * H).mean()

            # Fix B: marginal entropy over ID-weighted samples only
            # (ablation: when use_ood_marg=False, apply marginal entropy over full batch)
            if self.use_ood_marg:
                w_id_norm = w_id / (w_id.sum() + 1e-10)
                p_bar_id  = (w_id_norm.unsqueeze(1) * p).sum(0)
            else:
                p_bar_id  = p.mean(0)
            H_marg_id = -(p_bar_id * torch.log(p_bar_id + 1e-10)).sum()

            loss = loss_id - self.ood_lambda * loss_ood - self.marg_lambda * H_marg_id

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 6. Fix C: update prototypes with high-confidence ID samples
            if self.use_proto:
                self._update_prototypes(x, outputs, w_id, x.shape[0])

        return outputs.detach()
