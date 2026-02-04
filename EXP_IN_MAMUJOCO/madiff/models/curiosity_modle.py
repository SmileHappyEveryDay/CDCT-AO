# 新增好奇心模块文件: sc2/models/curiosity_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntrinsicCuriosityModule(nn.Module):
    """内在好奇心模块 (ICM) - 支持不同维度的输入输出"""
    def __init__(self, config):
        super().__init__()
        self.state_dim = config.local_obs_dim  
        self.action_dim = config.action_dim
        self.feature_dim = getattr(config, 'feature_dim', 128)
        
        # 特征提取网络 - 处理当前状态
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim)
        )
        
        # 下一状态特征提取网络 
        # 使用自适应输入层，根据实际输入调整
        self.next_feature_encoder = None  # 动态创建
        
        # 前向模型 (预测下一个状态特征)
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim)
        )
        
        # 逆向模型 (从状态变化预测动作)
        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
    
    def _create_next_encoder(self, next_obs_dim):
        """动态创建下一状态编码器"""
        if self.next_feature_encoder is None or self.next_feature_encoder[0].in_features != next_obs_dim:
            self.next_feature_encoder = nn.Sequential(
                nn.Linear(next_obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.feature_dim)
            )
            # 移动到正确的设备
            device = next(self.parameters()).device
            self.next_feature_encoder = self.next_feature_encoder.to(device)
    
    def forward(self, state, next_state, action):
        """
        Args:
            state: (batch_size, state_dim) - 当前状态
            next_state: (batch_size, next_state_dim) - 下一状态 (维度可能不同)
            action: (batch_size,) - 动作索引
        """
        # 确保输入在正确的设备上
        device = next(self.parameters()).device
        if state.device != device:
            state = state.to(device)
        if next_state.device != device:
            next_state = next_state.to(device)
        if action.device != device:
            action = action.to(device)
        
        # 动态创建下一状态编码器
        self._create_next_encoder(next_state.shape[-1])
        
        # 添加动作范围检查，防止one_hot编码崩溃
        action = torch.clamp(action.long(), 0, self.action_dim - 1)
        
        # 提取特征
        state_feat = self.feature_encoder(state)
        next_state_feat = self.next_feature_encoder(next_state)
        
        # 动作one-hot编码
        action_onehot = F.one_hot(action, self.action_dim).float()
        
        # 前向模型预测
        pred_next_feat = self.forward_model(
            torch.cat([state_feat, action_onehot], dim=-1)
        )
        
        # 逆向模型预测
        pred_action = self.inverse_model(
            torch.cat([state_feat, next_state_feat], dim=-1)
        )
        
        # 计算内在奖励 (预测误差)
        intrinsic_reward = F.mse_loss(
            pred_next_feat, next_state_feat.detach(), reduction='none'
        ).mean(dim=-1, keepdim=True)
        
        return intrinsic_reward, pred_action, pred_next_feat, next_state_feat