import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture
from tent import Tent
import torch.nn.functional as F


class RUniEnt(Tent):
    """
    R-UniEnt: Robust UniEnt for small-batch open-set TTA.

    Key improvements over UniEnt:
    1. Sliding-window score buffer: stabilises GMM fitting under small batch sizes
       by accumulating scores across batches before running EM.
    2. Piecewise adaptive weighting: reduces the influence of boundary samples
       (low GMM confidence) to avoid conflicting optimisation signals.
    3. Per-corruption model reset: prevents catastrophic drift when adapting
       sequentially across many corruption types.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False,
                 window_size=200, delta=0.6, alpha=0.5, warmup_steps=5):
        super().__init__(model, optimizer, steps, episodic)

        self.device = next(model.parameters()).device

        # ── source-domain class prototypes (classifier weights) ──────────────
        self.prototype = model.fc.weight.data.clone().to(self.device)

        # ── save a clean copy of model + optimiser for per-corruption reset ──
        self._model_state  = copy.deepcopy(model.state_dict())
        self._optim_state  = copy.deepcopy(optimizer.state_dict())

        # ── hook: capture block3 output (before bn1/relu/pool) ───────────────
        self._feat_buf = None
        def _hook(module, inp, out):
            self._feat_buf = out
        self.model.block3.register_forward_hook(_hook)

        # ── hyper-parameters ─────────────────────────────────────────────────
        self.window_size   = window_size   # sliding-window capacity
        self.delta         = delta         # high-confidence threshold
        self.alpha         = alpha         # boundary-sample weight decay
        self.warmup_steps  = warmup_steps  # batches before GMM is used

        # ── state ─────────────────────────────────────────────────────────────
        self.score_buffer  = []
        self.step          = 0

    # ── public reset (called between corruption types) ───────────────────────
    def reset_corruption(self):
        """Reset model weights, optimiser, buffer, and step counter."""
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optim_state)
        self.score_buffer = []
        self.step = 0

    # ── feature extraction ───────────────────────────────────────────────────
    def _extract_feat(self):
        """Convert raw block3 output → L2-normalised 128-d feature vector."""
        feat = F.relu(self.model.bn1(self._feat_buf))   # apply top BN+ReLU
        feat = F.avg_pool2d(feat, feat.shape[-1])        # global avg pool
        feat = feat.view(feat.size(0), -1)               # [B, 128]
        return F.normalize(feat, dim=1)

    # ── csOOD score ──────────────────────────────────────────────────────────
    def _csood_score(self, feat_norm):
        """
        1 - max_cosine_similarity(feat, prototypes).
        High score → likely OOD; low score → likely ID.
        """
        proto_norm = F.normalize(self.prototype, dim=1)
        cos_sim    = torch.matmul(feat_norm, proto_norm.T)   # [B, C]
        max_sim    = cos_sim.max(dim=1)[0]                   # [B]
        return (1.0 - max_sim).clamp(0.0, 1.0)

    # ── sliding-window min-max normalisation ─────────────────────────────────
    def _normalise(self, raw_score):
        if len(self.score_buffer) >= 50:
            buf = torch.tensor(
                self.score_buffer[-self.window_size:], device=self.device)
            s_min, s_max = buf.min(), buf.max()
        else:
            s_min, s_max = raw_score.min(), raw_score.max()
        return ((raw_score - s_min) / (s_max - s_min + 1e-8)).clamp(0.0, 1.0)

    # ── GMM-based sample weights ─────────────────────────────────────────────
    def _compute_weights(self, batch_scores_np):
        """
        Returns (w_id, w_ood) tensors for the current batch.
        Falls back to pure-Tent weights when GMM cannot distinguish ID/OOD.
        """
        B = len(batch_scores_np)
        ones  = torch.ones(B,  device=self.device)
        zeros = torch.zeros(B, device=self.device)

        # warmup: treat everything as ID (pure Tent behaviour)
        if self.step < self.warmup_steps:
            return ones, zeros

        # need enough buffer samples for a reliable GMM fit
        if len(self.score_buffer) < self.window_size:
            return ones, zeros

        scores = np.array(self.score_buffer)
        try:
            gmm = GaussianMixture(n_components=2, random_state=42, n_init=3,
                                  max_iter=100)
            gmm.fit(scores.reshape(-1, 1))

            # validity check: means must be sufficiently separated
            mean_diff = float(abs(gmm.means_[0] - gmm.means_[1]))
            if mean_diff < 0.15:
                return ones, zeros          # indistinguishable → pure Tent

            id_comp = int(np.argmin(gmm.means_.ravel()))
            # score the *current batch* (not the whole buffer)
            pi_id = gmm.predict_proba(
                batch_scores_np.reshape(-1, 1))[:, id_comp]
            pi_id = torch.tensor(pi_id, dtype=torch.float32,
                                 device=self.device)

            # piecewise adaptive weighting
            w_id = torch.where(
                pi_id >= self.delta,
                pi_id,
                torch.where(
                    pi_id <= 1.0 - self.delta,
                    zeros,
                    self.alpha * pi_id
                )
            )
            w_ood = torch.where(
                pi_id <= 1.0 - self.delta,
                1.0 - pi_id,
                torch.where(
                    pi_id >= self.delta,
                    zeros,
                    self.alpha * (1.0 - pi_id)
                )
            )
            return w_id, w_ood

        except Exception:
            return ones, zeros              # GMM failed → pure Tent

    # ── main forward ─────────────────────────────────────────────────────────
    @torch.enable_grad()
    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            # single forward pass (hook fills self._feat_buf)
            outputs = self.model(x)

            with torch.no_grad():
                feat_norm   = self._extract_feat()
                raw_score   = self._csood_score(feat_norm)
                csood_score = self._normalise(raw_score)

                # update sliding window
                self.score_buffer.extend(csood_score.cpu().numpy().tolist())
                if len(self.score_buffer) > self.window_size:
                    self.score_buffer = self.score_buffer[-self.window_size:]
                self.step += 1

                batch_scores_np = csood_score.cpu().numpy()
                w_id, w_ood = self._compute_weights(batch_scores_np)

            # ── unified entropy loss ──────────────────────────────────────
            p       = F.softmax(outputs, dim=1)
            H       = -(p * torch.log(p + 1e-10)).sum(dim=1)   # per-sample H

            loss_id  = (w_id  * H).mean()
            loss_ood = (w_ood * H).mean()

            p_bar    = p.mean(dim=0)
            H_marg   = -(p_bar * torch.log(p_bar + 1e-10)).sum()

            # λ1=0.5 (OOD entropy max), λ2=0.1 (marginal entropy, kept small)
            loss = loss_id - 0.5 * loss_ood - 0.1 * H_marg

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return outputs.detach()
