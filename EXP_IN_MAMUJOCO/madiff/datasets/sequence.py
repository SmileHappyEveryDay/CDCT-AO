import importlib
from typing import Callable, List, Optional

import numpy as np
import torch

from diffuser.datasets.buffer import ReplayBuffer
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.utils.mask_generator import MultiAgentMaskGenerator


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env_type: str = "d4rl",
        env: str = "hopper-medium-replay",
        n_agents: int = 2,
        horizon: int = 64,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        use_action: bool = True,
        discrete_action: bool = False,
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        termination_penalty: float = 0,
        use_padding: bool = True,
        discount: float = 0.99,
        returns_scale: float = 400.0,
        include_returns: bool = False, # 默认为 False
        include_env_ts: bool = False,
        history_horizon: int = 0,
        agent_share_parameters: bool = False,
        use_seed_dataset: bool = False,
        decentralized_execution: bool = False,
        use_inv_dyn: bool = True,
        use_zero_padding: bool = True,
        agent_condition_type: str = "single",
        pred_future_padding: bool = False,
        seed: Optional[int] = None,
    ):
        if use_seed_dataset:
            assert env_type == "mpe", f"Seed dataset only supported for MPE, not {env_type}"

        assert agent_condition_type in ["single", "all", "random"], agent_condition_type
        self.agent_condition_type = agent_condition_type

        env_mod_name = {
            "d4rl": "diffuser.datasets.d4rl",
            "mahalfcheetah": "diffuser.datasets.mahalfcheetah",
            "mamujoco": "diffuser.datasets.mamujoco",
            "mpe": "diffuser.datasets.mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[env_type]
        env_mod = importlib.import_module(env_mod_name)

        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = env_mod.load_environment(env)
        self.global_feats = env.metadata["global_feats"]

        self.use_inv_dyn = use_inv_dyn
        self.returns_scale = returns_scale
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.use_padding = use_padding
        self.use_action = use_action
        self.discrete_action = discrete_action
        # [!! 修改 !!] 强制关闭 returns
        self.include_returns = False 
        self.include_env_ts = include_env_ts
        self.decentralized_execution = decentralized_execution
        self.use_zero_padding = use_zero_padding
        self.pred_future_padding = pred_future_padding

        # 加载数据迭代器
        if env_type == "mpe" and use_seed_dataset:
            itr = env_mod.sequence_dataset(env, self.preprocess_fn, seed=seed)
        else:
            itr = env_mod.sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(
            n_agents,
            max_n_episodes,
            max_path_length,
            termination_penalty,
            global_feats=self.global_feats,
            use_zero_padding=self.use_zero_padding,
        )
        for _, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        # [!! 保留 !!] 针对 Ant 环境，强制切除最后 2 维全局坐标
        if 'observations' in fields.keys:
            obs = fields['observations']
            if obs.shape[-1] == 56:
                print(f"[SequenceDataset] 检测到 56 维观测数据 (Ant)，正在切除最后 2 维全局坐标...")
                fields['observations'] = obs[..., :-2]
                print(f"[SequenceDataset] 观测维度已修正为: {fields['observations'].shape}")

        # [!! 移除 !!] 移除了 RTG 预计算逻辑

        self.normalizer = DatasetNormalizer(
            fields,
            normalizer,
            path_lengths=fields["path_lengths"],
            agent_share_parameters=agent_share_parameters,
            global_feats=self.global_feats,
        )

        self.observation_dim = fields['observations'].shape[-1]
        self.action_dim = fields['actions'].shape[-1] if self.use_action else 0
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths

        self.indices = self.make_indices(fields.path_lengths)
        self.mask_generator = MultiAgentMaskGenerator(
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            history_horizon=self.history_horizon,
            action_visible=not use_inv_dyn,
        )

        if self.discrete_action:
            self.normalize(["observations"])
        else:
            self.normalize()

        self.pad_future()
        if self.history_horizon > 0:
            self.pad_history()

        print(fields)

    def pad_future(self, keys: List[str] = None):
        if keys is None:
            keys = ["normed_observations", "rewards", "terminals"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                keys.append("actions" if self.discrete_action else "normed_actions")
            # [!! 移除 !!] 不再添加 returns

        for key in keys:
            if key not in self.fields.keys: continue
            shape = self.fields[key].shape
            
            if self.use_zero_padding:
                self.fields[key] = np.concatenate(
                    [
                        self.fields[key],
                        np.zeros((shape[0], self.horizon - 1, *shape[2:]), dtype=self.fields[key].dtype),
                    ],
                    axis=1,
                )
            else:
                self.fields[key] = np.concatenate(
                    [
                        self.fields[key],
                        np.repeat(self.fields[key][:, -1:], self.horizon - 1, axis=1),
                    ],
                    axis=1,
                )

    def pad_history(self, keys: List[str] = None):
        if keys is None:
            keys = ["normed_observations", "rewards", "terminals"]
            if "legal_actions" in self.fields.keys:
                keys.append("legal_actions")
            if self.use_action:
                keys.append("actions" if self.discrete_action else "normed_actions")
            # [!! 移除 !!] 不再添加 returns

        for key in keys:
            if key not in self.fields.keys: continue
            shape = self.fields[key].shape
            if self.use_zero_padding:
                self.fields[key] = np.concatenate(
                    [
                        np.zeros((shape[0], self.history_horizon, *shape[2:]), dtype=self.fields[key].dtype),
                        self.fields[key],
                    ],
                    axis=1,
                )
            else:
                self.fields[key] = np.concatenate(
                    [
                        np.repeat(self.fields[key][:, :1], self.history_horizon, axis=1),
                        self.fields[key],
                    ],
                    axis=1,
                )

    def normalize(self, keys: List[str] = None):
        if keys is None:
            keys = ["observations", "actions"] if self.use_action else ["observations"]
        for key in keys:
            shape = self.fields[key].shape
            array = self.fields[key].reshape(shape[0] * shape[1], *shape[2:])
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(shape)

    def make_indices(self, path_lengths: np.ndarray):
        indices = []
        for i, path_length in enumerate(path_lengths):
            if self.use_padding:
                max_start = path_length - 1
            else:
                max_start = path_length - self.horizon
                if max_start < 0: continue
            for start in range(max_start):
                end = start + self.horizon
                mask_end = min(end, path_length)
                indices.append((i, start, end, mask_end))
        return np.array(indices)

    def get_conditions(self, observations: np.ndarray, agent_idx: Optional[int] = None):
        ret_dict = {}
        if self.agent_condition_type == "single":
            cond_observations = np.zeros_like(observations[: self.history_horizon + 1])
            cond_observations[:, agent_idx] = observations[
                : self.history_horizon + 1, agent_idx
            ]
            ret_dict["agent_idx"] = torch.LongTensor([[agent_idx]])
        elif self.agent_condition_type == "all":
            cond_observations = observations[: self.history_horizon + 1]
        ret_dict[(0, self.history_horizon + 1)] = cond_observations
        return ret_dict

    def __len__(self):
        if self.agent_condition_type == "single":
            return len(self.indices) * self.n_agents
        else:
            return len(self.indices)

    def __getitem__(self, idx: int):
        if self.agent_condition_type == "single":
            path_ind, start, end, mask_end = self.indices[idx // self.n_agents]
            agent_mask = np.zeros(self.n_agents, dtype=bool)
            agent_mask[idx % self.n_agents] = 1
        elif self.agent_condition_type == "all":
            path_ind, start, end, mask_end = self.indices[idx]
            agent_mask = np.ones(self.n_agents, dtype=bool)
        elif self.agent_condition_type == "random":
            path_ind, start, end, mask_end = self.indices[idx]
            agent_mask = np.random.randint(0, 2, self.n_agents, dtype=bool)

        history_start = start
        start = history_start + self.history_horizon
        end = end + self.history_horizon
        mask_end = mask_end + self.history_horizon

        # [!! 保留 !!] 强制修正切片长度的逻辑
        observations = self.fields.normed_observations[path_ind, history_start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self.fields.actions[path_ind, history_start:end]
            else:
                actions = self.fields.normed_actions[path_ind, history_start:end]

        # [!! 保留 !!] 检查 observations 和 actions 的长度
        current_len = observations.shape[0]
        if current_len < self.horizon:
            pad_len = self.horizon - current_len
            obs_padding = np.zeros((pad_len, *observations.shape[1:]), dtype=observations.dtype)
            observations = np.concatenate([observations, obs_padding], axis=0)
            if self.use_action:
                act_padding = np.zeros((pad_len, *actions.shape[1:]), dtype=actions.dtype)
                actions = np.concatenate([actions, act_padding], axis=0)

        if self.use_action:
            # [!! 保留 !!] 交换拼接顺序：先 observations，后 actions
            trajectories = np.concatenate([observations, actions], axis=-1)
        else:
            trajectories = observations
            
        if trajectories.shape[0] != self.horizon:
             trajectories = trajectories[:self.horizon]

        if self.use_inv_dyn:
            cond_masks = self.mask_generator(observations.shape, agent_mask)
            cond_trajectories = observations.copy()
        else:
            cond_masks = self.mask_generator(trajectories.shape, agent_mask)
            cond_trajectories = trajectories.copy()
        cond_trajectories[: self.history_horizon, ~agent_mask] = 0.0
        
        cond = {"x": cond_trajectories, "masks": cond_masks}

        loss_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        if self.pred_future_padding:
            loss_masks[self.history_horizon :] = 1.0
        else:
            loss_masks[self.history_horizon : mask_end - history_start] = 1.0
        if self.use_inv_dyn:
            loss_masks[self.history_horizon, agent_mask] = 0.0

        attention_masks = np.zeros((observations.shape[0], observations.shape[1], 1))
        attention_masks[self.history_horizon : mask_end - history_start] = 1.0
        attention_masks[: self.history_horizon, agent_mask] = 1.0

        batch = {
            "x": trajectories,
            "cond": cond,
            "loss_masks": loss_masks,
            "attention_masks": attention_masks,
        }

        # [!! 移除 !!] 移除了 returns 的获取逻辑

        if self.include_env_ts:
            env_ts = np.arange(history_start, start + self.horizon) - self.history_horizon
            env_ts[np.where(env_ts < 0)] = self.max_path_length
            env_ts[np.where(env_ts >= self.max_path_length)] = self.max_path_length
            batch["env_ts"] = env_ts

        if "legal_actions" in self.fields.keys:
            batch["legal_actions"] = self.fields.legal_actions[path_ind, history_start:end]

        return batch

class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.include_returns is True

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        value_batch = {
            "x": batch["x"],
            "cond": batch["cond"],
            "returns": batch["returns"].mean(axis=-1),
        }
        return value_batch


class BCSequenceDataset(SequenceDataset):
    def __init__(
        self,
        env_type: str = "d4rl",
        env: str = "hopper-medium-replay",
        n_agents: int = 2,
        normalizer: str = "LimitsNormalizer",
        preprocess_fns: List[Callable] = [],
        max_path_length: int = 1000,
        max_n_episodes: int = 10000,
        agent_share_parameters: bool = False,
    ):
        super().__init__(
            env_type=env_type,
            env=env,
            n_agents=n_agents,
            normalizer=normalizer,
            preprocess_fns=preprocess_fns,
            max_path_length=max_path_length,
            max_n_episodes=max_n_episodes,
            agent_share_parameters=agent_share_parameters,
            horizon=1,
            history_horizon=0,
            use_action=True,
            termination_penalty=0.0,
            use_padding=False,
            discount=1.0,
            include_returns=False,
        )

    def __getitem__(self, idx: int):
        path_ind, start, end, _ = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        batch = {"observations": observations, "actions": actions}
        return batch
