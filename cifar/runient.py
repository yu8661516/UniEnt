import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture
from tent import Tent
import torch.nn.functional as F

class RUniEnt(Tent):
    def __init__(self, model, optimizer, steps=1, episodic=False, 
                 window_size=500, delta=0.7, alpha=0.3, warmup_steps=10):
        super().__init__(model, optimizer, steps, episodic)
        
        self.device = next(model.parameters()).device
        
        # 保存源域分类器权重作为原型
        self.prototype = model.fc.weight.data.clone().to(self.device)
        
        # 注册forward hook抓取特征
        self.features = None
        def hook(module, input, output):
            self.features = output
        self.model.block3.register_forward_hook(hook)
        
        # ✅ 优化后的超参数（针对CIFAR-10-C重度偏移）
        self.window_size = window_size  # 增大窗口，提高GMM稳定性
        self.delta = delta              # 提高阈值，只对高置信度样本加权
        self.alpha = alpha              # 降低中间区域权重，减少噪声影响
        self.warmup_steps = warmup_steps  # 延长warmup，积累更多样本
        
        self.score_buffer = []
        self.step = 0

    @torch.enable_grad()
    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            # 一次前向传播，同时得到logits和特征
            outputs = self.model(x)
            
            # 提取特征（与原模型完全一致）
            feat = F.relu(self.model.bn1(self.features))
            feat = F.avg_pool2d(feat, 8)
            feat = feat.view(-1, self.model.nChannels)
            
            with torch.no_grad():
                # 计算csOOD评分
                feat_norm = F.normalize(feat, dim=1)
                proto_norm = F.normalize(self.prototype, dim=1)
                cos_sim = torch.matmul(feat_norm, proto_norm.T)
                max_sim = torch.max(cos_sim, dim=1)[0]
                raw_score = 1 - max_sim
                
                # 滑动窗口归一化
                if len(self.score_buffer) >= 100:
                    buffer_tensor = torch.tensor(self.score_buffer[-self.window_size:], device=self.device)
                    score_min = buffer_tensor.min()
                    score_max = buffer_tensor.max()
                    csOOD_score = (raw_score - score_min) / (score_max - score_min + 1e-10)
                else:
                    csOOD_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-10)
                
                csOOD_score = csOOD_score.clamp(0, 1)
                self.score_buffer.extend(csOOD_score.cpu().numpy().tolist())
                if len(self.score_buffer) > self.window_size:
                    self.score_buffer = self.score_buffer[-self.window_size:]
                
                # ✅ 改进的GMM拟合：增加拟合失败的降级策略
                self.step += 1
                if self.step < self.warmup_steps:
                    # warmup阶段：退化为纯Tent，不做区分
                    w_id = torch.ones(len(x), device=self.device)
                    w_ood = torch.zeros(len(x), device=self.device)
                else:
                    if len(self.score_buffer) >= self.window_size:
                        scores = np.array(self.score_buffer)
                        try:
                            gmm = GaussianMixture(n_components=2, random_state=42, n_init=5)
                            gmm.fit(scores.reshape(-1, 1))
                            
                            # ✅ 增加聚类有效性检验：如果两个均值差小于0.2，说明无法区分
                            mean_diff = abs(gmm.means_[0] - gmm.means_[1])
                            if mean_diff < 0.2:
                                # 无法区分ID/OOD，退化为纯Tent
                                w_id = torch.ones(len(x), device=self.device)
                                w_ood = torch.zeros(len(x), device=self.device)
                            else:
                                id_comp = np.argmin(gmm.means_)
                                pi_id = gmm.predict_proba(scores.reshape(-1, 1))[:, id_comp]
                                pi_id = pi_id[-len(x):]
                                pi_id = torch.tensor(pi_id, device=self.device)
                                
                                # ✅ 恢复原论文的分段加权逻辑（去掉有害的clamp）
                                w_id = torch.where(
                                    pi_id >= self.delta,
                                    pi_id,
                                    torch.where(
                                        pi_id <= 1 - self.delta,
                                        torch.zeros_like(pi_id),
                                        self.alpha * pi_id
                                    )
                                )
                                w_ood = torch.where(
                                    pi_id <= 1 - self.delta,
                                    1 - pi_id,
                                    torch.where(
                                        pi_id >= self.delta,
                                        torch.zeros_like(pi_id),
                                        self.alpha * (1 - pi_id)
                                    )
                                )
                        except:
                            # GMM拟合失败，退化为纯Tent
                            w_id = torch.ones(len(x), device=self.device)
                            w_ood = torch.zeros(len(x), device=self.device)
                    else:
                        # 样本不足，退化为纯Tent
                        w_id = torch.ones(len(x), device=self.device)
                        w_ood = torch.zeros(len(x), device=self.device)

            # 计算R-UniEnt损失
            p = F.softmax(outputs, dim=1)
            entropy = -torch.sum(p * torch.log(p + 1e-10), dim=1)
            
            loss_id = (w_id * entropy).mean()
            loss_ood = (w_ood * entropy).mean()
            p_bar = p.mean(dim=0)
            marginal_entropy = -torch.sum(p_bar * torch.log(p_bar + 1e-10))
            
            # ✅ 调整损失系数：降低边缘熵的权重，防止模型坍缩
            loss = loss_id - 0.5 * loss_ood - 0.3 * marginal_entropy
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return outputs.detach()
