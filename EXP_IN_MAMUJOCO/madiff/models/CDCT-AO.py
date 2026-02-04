import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# ====================================================================
#  Helper Classes and Functions
# ====================================================================

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, state_dim, act_dim, block_size, **kwargs):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # [!! 修改 !!] Mask 大小改为 block_size * 2 (因为只有 state 和 action)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size * 2, config.block_size * 2))
                                     .view(1, 1, config.block_size * 2, config.block_size * 2))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ====================================================================
#  第一部分：纯粹的 GPT 模型
# ====================================================================

class GPT(nn.Module):
    """ a vanilla GPT model, with a single head that returns the same number of features as the input """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # [!! 修改 !!] 位置编码长度改为 block_size * 2
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size * 2, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.act_dim, bias=False)

        self.state_encoder = nn.Linear(config.state_dim, config.n_embd)
        self.action_encoder = nn.Linear(config.act_dim, config.n_embd)
        # [!! 移除 !!] 移除了 ret_emb

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # [!! 修改 !!] 移除了 rtgs 参数
    def forward(self, states, actions, timesteps):
        B, T, N, D_s = states.shape
        
        # [!! 关键修复 !!] 
        # 原始错误逻辑: states = states.reshape(B * N, T, D_s)
        # 导致时间维度 T 和智能体维度 N 混淆。
        # 正确逻辑: 先交换维度 (B, T, N, D) -> (B, N, T, D)，再合并 B*N
        
        states = states.transpose(1, 2).reshape(B * N, T, D_s)
        actions = actions.transpose(1, 2).reshape(B * N, T, self.config.act_dim)
        
        timesteps = timesteps.unsqueeze(2).repeat(1, 1, N).transpose(1, 2).reshape(B * N, T)

        state_embeddings = self.state_encoder(states)
        action_embeddings = self.action_encoder(actions)

        token_embeddings = torch.zeros((B * N, T * 2, self.config.n_embd), dtype=torch.float32, device=states.device)
        token_embeddings[:, ::2, :] = state_embeddings
        token_embeddings[:, 1::2, :] = action_embeddings

        t_global = timesteps.view(B * N, -1, 1).repeat(1, 1, self.config.n_embd)
        global_pos_emb = torch.gather(self.global_pos_emb.expand(B * N, -1, -1), 1, t_global)
        position_embeddings = self.pos_emb[:, :T*2, :] + global_pos_emb.repeat_interleave(2, dim=1)

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # [!! 核心修改 !!]
        # 我们只关心对序列中最后一个状态 s_{T-1} 的动作预测。
        preds = logits[:, -2, :]  # Shape: (B*N, ActDim)
        
        # [!! 正确的修复 !!]
        # 目标形状: (B, T, N, ActDim)，其中 T=1
        # 1. 还原 B 和 N: (B*N, ActDim) -> (B, N, ActDim)
        preds = preds.reshape(B, N, self.config.act_dim)
        
        # 2. 交换 N 和 T (T=1): (B, N, ActDim) -> (B, ActDim, N)
        #    然后增加 T 维度: (B, ActDim, N) -> (B, 1, ActDim, N)
        #    最后交换 N 和 D: (B, 1, ActDim, N) -> (B, 1, N, ActDim)
        #    这个过程太复杂了。我们换一种更直接的方式。

        # [!! 更简洁、正确的修复 !!]
        # 1. 还原 B 和 N: (B*N, ActDim) -> (B, N, ActDim)
        preds = preds.reshape(B, N, self.config.act_dim)
        
        # 2. 在时间维度上增加一个维度: (B, N, ActDim) -> (B, N, 1, ActDim)
        preds = preds.unsqueeze(2)
        
        # 3. 交换时间和智能体维度以匹配 (B, T, N, D) 约定:
        #    (B, N, 1, ActDim) -> (B, 1, N, ActDim)
        preds = preds.transpose(1, 2)
        
        return preds

# ====================================================================
#  第二部分：MADC_BC 包装器
# ====================================================================

class MADC_BC(nn.Module):
    """
    MADC model for Behavior Cloning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GPT(config)

    # [!! 修改 !!] 修正参数接收：env_ts 和 loss_masks 是 batch 的顶层键，不在 cond 中
    def loss(self, x, cond, env_ts=None, loss_masks=None, **kwargs):
        state_dim = self.config.state_dim
        act_dim = self.config.act_dim
        
        states = x[..., :state_dim]
        actions = x[..., state_dim : state_dim + act_dim]

        # 兼容性处理：如果未通过参数传入，尝试从 kwargs 获取
        if env_ts is None:
            env_ts = kwargs.get('env_ts')
        if loss_masks is None:
            loss_masks = kwargs.get('loss_masks')

        if env_ts is None:
            raise ValueError("env_ts is required for MADC loss calculation")
        if loss_masks is None:
            raise ValueError("loss_masks is required for MADC loss calculation")

        # action_preds shape: (Batch, 1, Agents, ActDim)
        action_preds = self.model(
            states,
            actions,
            env_ts,
        )

        # [!! 核心修改 !!]
        # 目标动作也只取序列中的最后一个
        target_actions = actions[:, -1, :, :] # Shape: (Batch, Agents, ActDim)
        
        # [!! 核心修改 !!]
        # 预测的动作只有一个时间步，所以展平即可
        # action_preds shape: (B, 1, N, D) -> (B, N, D)
        flat_preds = action_preds.squeeze(1).reshape(-1, act_dim)
        flat_targets = target_actions.reshape(-1, act_dim)
        
        # 由于我们只对最后一个时间步计算损失，不再需要复杂的 loss_mask
        loss = torch.mean((flat_preds - flat_targets) ** 2)
        infos = {"action_mse_loss": loss.item()}
        return loss, infos

    def get_action(self, states, actions, timesteps, **kwargs):
        with torch.no_grad():
            action_preds = self.model(
                states=states,
                actions=actions,
                timesteps=timesteps,
            )
        return action_preds