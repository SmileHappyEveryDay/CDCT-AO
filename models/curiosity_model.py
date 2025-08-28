# 新增好奇心模块文件: sc2/models/curiosity_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntrinsicCuriosityModule(nn.Module):
    """Curiosity Module, supports different input/output dimensions"""
    def __init__(self, config):
        super().__init__()
        self.state_dim = config.local_obs_dim  
        self.action_dim = config.action_dim
        self.feature_dim = getattr(config, 'feature_dim', 128)
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim)
        )
        self.next_feature_encoder = None  
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
    
    def _create_next_encoder(self, next_obs_dim):
        if self.next_feature_encoder is None or self.next_feature_encoder[0].in_features != next_obs_dim:
            self.next_feature_encoder = nn.Sequential(
                nn.Linear(next_obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.feature_dim)
            )
            device = next(self.parameters()).device
            self.next_feature_encoder = self.next_feature_encoder.to(device)
    
    def forward(self, state, next_state, action):
        """
        Args:
            state: (batch_size, state_dim)
            next_state: (batch_size, next_state_dim)
            action: (batch_size, action_dim)
        """

        device = next(self.parameters()).device
        if state.device != device:
            state = state.to(device)
        if next_state.device != device:
            next_state = next_state.to(device)
        if action.device != device:
            action = action.to(device)
        self._create_next_encoder(next_state.shape[-1])
        action = torch.clamp(action.long(), 0, self.action_dim - 1)
        state_feat = self.feature_encoder(state)
        next_state_feat = self.next_feature_encoder(next_state)
        action_onehot = F.one_hot(action, self.action_dim).float()
        pred_next_feat = self.forward_model(
            torch.cat([state_feat, action_onehot], dim=-1)
        )
        pred_action = self.inverse_model(
            torch.cat([state_feat, next_state_feat], dim=-1)
        )
        intrinsic_reward = F.mse_loss(
            pred_next_feat, next_state_feat.detach(), reduction='none'
        ).mean(dim=-1, keepdim=True)
        
        return intrinsic_reward, pred_action, pred_next_feat, next_state_feat