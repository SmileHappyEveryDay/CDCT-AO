import os

# [!! Modified !!] Removed complex rendering setup since video is disabled
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

import gc
import multiprocessing
import pickle
import sys
import time
from copy import deepcopy
from collections import deque
from multiprocessing import Pipe, connection
from multiprocessing.context import Process
import pprint

import numpy as np
import torch
from ml_logger import logger
# [!! Modified !!] Removed imageio import
# import imageio 

import diffuser.utils as utils
from diffuser.utils.arrays import to_np, to_torch
from diffuser.utils.launcher_util import dict_to_config


class MADCEvaluatorWorker(Process):
    def __init__(
        self,
        parent_remote: connection.Connection,
        child_remote: connection.Connection,
        queue: multiprocessing.Queue,
        verbose: bool = False,
    ):
        self.parent_remote = parent_remote
        self.p = child_remote
        self.queue = queue
        self.initialized = False
        self.verbose = verbose
        super().__init__()

    def _evaluate(self, load_step=None):
        assert (
            self.initialized is True
        ), "Evaluator should be initialized before evaluation."

        Config = self.Config
        loadpath = os.path.join(self.log_dir, "checkpoint")
        
        results_step_id = "latest"
        if load_step is not None:
            loadpath = os.path.join(loadpath, f"state_{load_step}.pt")
            results_step_id = load_step
        else:
            loadpath = os.path.join(loadpath, "state.pt")

        print(f"[INFO] 正在加载训练好的模型权重: {loadpath}")

        state_dict = torch.load(loadpath, map_location='cpu')
        self.trainer.step = state_dict["step"]
        self.trainer.model.load_state_dict(state_dict["model"])
        
        self.trainer.model.to(Config.device)
        self.trainer.model.eval()

        num_eval = Config.num_eval
        device = Config.device

        dones = [0 for _ in range(num_eval)]
        episode_rewards = [np.zeros(Config.n_agents) for _ in range(num_eval)]

        history_len = Config.horizon - 1
        
        # [!! Modified !!] Removed video saving setup

        obs_list_raw = [env.reset() for env in self.env_list]
        
        print(f"[INFO] 已启动 {len(self.env_list)} 个评估环境。")
        print(f"[INFO] 每个环境最大步数 (max_path_length): {Config.max_path_length}")
        
        # [!! Modified !!] Removed pre-render check

        # 检查是否为 (obs, info) 元组
        first_ret = obs_list_raw[0]
        if isinstance(first_ret, tuple) and len(first_ret) == 2 and isinstance(first_ret[1], dict):
            real_obs_list = [ret[0] for ret in obs_list_raw]
        else:
            real_obs_list = obs_list_raw
            
        try:
            obs = np.stack([np.stack(o) for o in real_obs_list])
        except Exception as e:
            print(f"[ERROR] 观测值堆叠失败: {e}")
            raise e

        # 维度适配与填充
        # [!! 插入修复代码 !!]
        # ------------------------------------------------------------------
        # 1. 预处理：确保与训练时的 SequenceDataset 逻辑一致
        # 训练时: if obs.shape[-1] == 56: obs = obs[..., :-2]
        # ------------------------------------------------------------------
        if obs.shape[-1] == 56 and self.Config.state_dim == 54:
            # print("[DEBUG] 切除 Ant 环境最后 2 维全局坐标 (56 -> 54)") # 调试时可打开
            obs = obs[..., :-2]
        
        # 2. 维度检查与适配 (原有逻辑)
        env_dim = obs.shape[-1]
        target_dim = self.Config.state_dim 

        if env_dim != target_dim:
            # 如果切片后仍然不匹配，说明配置有更严重的问题
            print(f"[Warning] 观测维度不匹配! Env: {env_dim}, Model: {target_dim}")
            # ...existing code... (原有的 padding 逻辑)
        
        state_queues = [deque(maxlen=history_len) for _ in range(num_eval)]
        action_queues = [deque(maxlen=history_len) for _ in range(num_eval)]
        
        zero_state = np.zeros_like(self.normalizer.normalize(obs[0], "observations"))
        zero_action = np.zeros((Config.n_agents, Config.act_dim))
        for i in range(num_eval):
            for _ in range(history_len):
                state_queues[i].append(zero_state)
                action_queues[i].append(zero_action)

        returns = to_torch(Config.returns_scale * torch.ones(num_eval, Config.n_agents, 1), device=device)
        
        t = 0
        while sum(dones) < num_eval:
            
            if obs.shape[-1] != target_dim:
                 current_env_dim = obs.shape[-1]
                 missing_dim = target_dim - current_env_dim
                 if missing_dim == 2:
                     global_pos = get_global_pos(self.env_list)
                     fill_chunk = np.tile(global_pos[:, None, :], (1, Config.n_agents, 1))
                 else:
                     fill_chunk = np.zeros((num_eval, Config.n_agents, missing_dim), dtype=obs.dtype)
                 obs = np.concatenate([obs, fill_chunk], axis=-1)

            current_obs_norm = self.normalizer.normalize(obs, "observations")
            
            if t == 0:
                print(f"[DEBUG] Step 0 Obs Norm Mean: {np.mean(current_obs_norm):.4f}, Min: {np.min(current_obs_norm):.4f}, Max: {np.max(current_obs_norm):.4f}")
            
            batch_states_hist = to_torch(np.stack([list(q) for q in state_queues]), device=device)
            batch_actions_hist = to_torch(np.stack([list(q) for q in action_queues]), device=device)
            
            # 1. 构造完整的状态序列 (History + Current) -> Length: Horizon
            input_states = torch.cat([batch_states_hist, to_torch(current_obs_norm, device=device).unsqueeze(1)], dim=1)
            
            # 2. [修复] 构造完整的动作序列 (History + Dummy) -> Length: Horizon
            # 我们需要为当前步 t 提供一个占位动作 (全0)，虽然 GPT 预测 a_t 时不会用到它(被Mask)，但输入形状必须匹配
            dummy_action = torch.zeros((num_eval, 1, Config.n_agents, Config.act_dim), device=device)
            input_actions = torch.cat([batch_actions_hist, dummy_action], dim=1)

            # 3. [修复] 修正时间步逻辑
            # 窗口应该是 [t - (Horizon-1), ..., t]
            # 例如 Horizon=20, t=100. 窗口是 81...100.
            # 之前的代码是 arange(t, t+H)，那是未来的时间步，完全错误。
            start_t = t - history_len
            timesteps = torch.arange(start_t, start_t + history_len + 1, device=device).unsqueeze(0).repeat(num_eval, 1)
            
            # 限制 timesteps 范围 (处理 t < history_len 时的负数，以及最大步数限制)
            max_emb_idx = 999
            if hasattr(self.trainer.model.model, 'embed_timestep') and hasattr(self.trainer.model.model.embed_timestep, 'num_embeddings'):
                max_emb_idx = self.trainer.model.model.embed_timestep.num_embeddings - 1
            elif hasattr(Config, 'max_timestep'):
                max_emb_idx = Config.max_timestep - 1
            
            timesteps = torch.clamp(timesteps, min=0, max=max_emb_idx)

            with torch.no_grad():
                # [修复] 传入完整的序列，不再切片 [:-1]
                pred_actions = self.trainer.model.model(
                    input_states,
                    input_actions,
                    timesteps
                )
                # 取最后一个时间步的预测结果 (对应于 current_obs_norm)
                action_normed = pred_actions[:, -1, :, :]

            action_normed = to_np(action_normed)
            
            # [!! 关键修复 !!] 反归一化动作
            action_real = self.normalizer.unnormalize(action_normed, 'actions')
            
            obs_list_new = []
            for i in range(num_eval):
                state_queues[i].append(current_obs_norm[i])
                action_queues[i].append(action_normed[i])

                # 执行动作
                step_ret = self.env_list[i].step(action_real[i])
                
                # 适配不同的返回值结构
                if len(step_ret) == 4:
                    this_obs, this_reward, this_done, this_info = step_ret
                elif len(step_ret) == 5:
                    this_obs, this_reward, terminated, truncated, this_info = step_ret
                    this_done = terminated or truncated
                elif len(step_ret) == 3:
                    this_reward, this_done, this_info = step_ret
                    # 尝试手动获取 obs
                    try:
                        if hasattr(self.env_list[i], 'get_obs'):
                            this_obs = self.env_list[i].get_obs()
                        elif hasattr(self.env_list[i], '_get_obs'):
                            this_obs = self.env_list[i]._get_obs()
                        elif hasattr(self.env_list[i], 'unwrapped') and hasattr(self.env_list[i].unwrapped, '_get_obs'):
                            this_obs = self.env_list[i].unwrapped._get_obs()
                        else:
                            if t == 0 and i == 0:
                                print("[WARN] 无法找到 get_obs 方法，使用全零观测值填充！")
                            this_obs = np.zeros((Config.n_agents, self.Config.state_dim))
                    except Exception as e:
                        print(f"[ERROR] 获取观测值失败: {e}")
                        this_obs = np.zeros((Config.n_agents, self.Config.state_dim))
                else:
                    raise ValueError(f"Unexpected step return length: {len(step_ret)}")

                # [!! 调试 !!] 详细打印第一个环境的前 10 步和每 100 步的数据
                if i == 0 and (t < 10 or t % 100 == 0):
                    print(f"\n--- [DEBUG] Env 0 Step {t} ---")
                    print(f"Action (Real): {action_real[i]}")
                    print(f"Reward: {this_reward}")
                    print(f"Done: {this_done}")
                    print(f"Info: {this_info}")
                    if hasattr(self.env_list[i], 'get_state'):
                         # 尝试打印一些状态信息，如位置
                         pass

                if isinstance(this_obs, list):
                    this_obs = np.stack(this_obs)
                
                obs_list_new.append(this_obs[None])
                
                # 处理多智能体 done (可能是列表或布尔值)
                if hasattr(this_done, 'all'):
                    is_done = this_done.all()
                elif isinstance(this_done, list) or isinstance(this_done, np.ndarray):
                    is_done = np.all(this_done)
                else:
                    is_done = this_done

                # [!! 临时 !!] 强制忽略 Done 信号，看看能不能跑起来
                # 如果是因为健康检查挂掉，强制继续可以验证动作是否有效
                # is_done = False 

                if is_done or t >= Config.max_path_length - 1:
                    if dones[i] == 0:
                        dones[i] = 1
                        # 确保 reward 是数值
                        try:
                            r_sum = float(episode_rewards[i].sum()) + float(np.sum(this_reward))
                        except:
                            r_sum = 0.0
                        episode_rewards[i] += this_reward
                        logger.print(f"Episode ({i}) finished at step {t+1} with reward: {r_sum}", color="green")
                else:
                    if dones[i] == 0:
                        episode_rewards[i] += this_reward

            obs = np.concatenate(obs_list_new, axis=0)
            t += 1

        # [!! Modified !!] Removed video saving block

        episode_rewards = np.array([r.sum() for r in episode_rewards])
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        logger.print(f"average_ep_reward: {avg_reward}, std_ep_reward: {std_reward}", color="green")
        
        results_path = f"results/step_{results_step_id}-ep_{num_eval}.json"
        logger.save_json(
            {
                "average_ep_reward": avg_reward.tolist(),
                "std_ep_reward": std_reward.tolist(),
            },
            results_path,
        )
        
        self.p.send("evaluation_done")

    def _init(self, log_dir, device, num_eval=10, **kwargs):
        self.log_dir = log_dir
        self.device = device
        self.num_eval = num_eval # 保存 num_eval 以便后续使用

        print("[DEBUG] _init: 开始初始化...")
        assert self.initialized is False, "Evaluator can only be initialized once."

        with open(os.path.join(log_dir, "parameters.pkl"), "rb") as f:
            params = pickle.load(f)

        config_dict = dict(params['Config'])
        config_dict.update(kwargs)
        
        # [!! Modified !!] Reset to 10 environments (or whatever config says, default 10)
        # Removed the forced upgrade to 50
        if 'num_eval' not in config_dict:
            config_dict['num_eval'] = 10
        
        if 'block_size' not in config_dict and 'horizon' in config_dict:
            config_dict['block_size'] = config_dict['horizon']

        Config = dict_to_config(config_dict)
        self.Config = Config
        self.Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.configure(log_dir, "evaluation_logs")
        torch.backends.cudnn.benchmark = True

        print("[DEBUG] _init: 加载数据集...")
        dataset_config = utils.Config(
            Config.loader,
            horizon=Config.horizon,
            env=Config.dataset,
            use_padding=Config.use_padding,
            max_path_length=Config.max_path_length,
            env_type=Config.env_type,
            preprocess_fns=[], 
        )
        dataset = dataset_config()
        self.normalizer = dataset.normalizer
        
        # Normalizer 修复逻辑 (保持不变)
        print(f"\n{'='*20} Normalizer 检查 {'='*20}")
        try:
            if hasattr(self.normalizer, 'normalizers'):
                for key, agent_norms in self.normalizer.normalizers.items():
                    for i, norm in enumerate(agent_norms):
                        if hasattr(norm, 'maxs') and hasattr(norm, 'mins'):
                            ranges = norm.maxs - norm.mins
                            problem_indices = np.where(ranges < 1e-4)[0]
                            if len(problem_indices) > 0:
                                print(f"[WARN] Agent {i} Key {key}: 维度 {problem_indices} 的范围极小，强制修复。")
                                norm.maxs[problem_indices] += 1.0
                        if hasattr(norm, 'std'):
                            problem_indices = np.where(norm.std < 1e-4)[0]
                            if len(problem_indices) > 0:
                                print(f"[WARN] Agent {i} Key {key}: 维度 {problem_indices} 的标准差极小，强制修复。")
                                norm.std[problem_indices] = 1.0
        except Exception as e:
            print(f"[WARN] Normalizer 检查/修复失败: {e}")
        print(f"{'='*50}\n")
        
        if hasattr(dataset, 'state_dim'):
            self.Config.state_dim = dataset.state_dim
        elif hasattr(dataset, 'observation_dim'):
            self.Config.state_dim = dataset.observation_dim
        elif hasattr(dataset, 'fields'):
            self.Config.state_dim = dataset.fields['observations'].shape[-1]
        else:
            raise AttributeError("Cannot find state_dim or observation_dim in dataset")

        if hasattr(dataset, 'act_dim'):
            self.Config.act_dim = dataset.act_dim
        elif hasattr(dataset, 'action_dim'):
            self.Config.act_dim = dataset.action_dim
        elif hasattr(dataset, 'fields'):
            self.Config.act_dim = dataset.fields['actions'].shape[-1]
        else:
            raise AttributeError("Cannot find act_dim or action_dim in dataset")

        print("[DEBUG] _init: 检测模型类...")
        try:
            import diffuser.models.madc as madc_module
            candidates = [name for name in dir(madc_module) if 'MADC' in name and isinstance(getattr(madc_module, name), type)]
            
            if candidates:
                class_name = 'MADC' if 'MADC' in candidates else candidates[0]
                model_class_path = f'diffuser.models.madc.{class_name}'
                print(f"[INFO] 自动检测到模型类名为: {class_name}")
            else:
                model_class_path = 'diffuser.models.madc.MADC'

            if 'model' in Config and 'MADC_BC' in Config.model:
                 model_class_path = Config.model
        except Exception as e:
            print(f"[ERROR] 模型类检测失败: {e}")
            raise e
        
        class SafeConfig:
            def __init__(self, **entries):
                self.__dict__.update(entries)
            def __repr__(self):
                return str(self.__dict__)

        safe_config_dict = {}
        if hasattr(Config, '__dict__'):
            safe_config_dict.update(Config.__dict__)
        elif isinstance(Config, dict):
            safe_config_dict.update(Config)
        
        critical_attrs = [
            'device', 'state_dim', 'act_dim', 'block_size', 'horizon', 
            'n_embd', 'n_layer', 'n_head', 'max_path_length',
            'embd_pdrop', 'resid_pdrop', 'attn_pdrop'
        ]
        for attr in critical_attrs:
            if hasattr(Config, attr):
                safe_config_dict[attr] = getattr(Config, attr)
        
        if 'block_size' not in safe_config_dict and 'horizon' in safe_config_dict:
             safe_config_dict['block_size'] = safe_config_dict['horizon']

        if 'max_timestep' not in safe_config_dict:
             if 'max_path_length' in safe_config_dict:
                 safe_config_dict['max_timestep'] = safe_config_dict['max_path_length']
             else:
                 safe_config_dict['max_timestep'] = 1000

        for pdrop in ['embd_pdrop', 'resid_pdrop', 'attn_pdrop']:
            if pdrop not in safe_config_dict:
                safe_config_dict[pdrop] = 0.1

        safe_config = SafeConfig(**safe_config_dict)

        print("[DEBUG] _init: 实例化模型...")
        try:
            import inspect
            model_cls = utils.config.import_class(model_class_path)
            sig = inspect.signature(model_cls.__init__)
            
            potential_args = {
                'n_embd': getattr(Config, 'n_embd', None),
                'n_head': getattr(Config, 'n_head', None),
                'n_layer': getattr(Config, 'n_layer', None),
                'horizon': Config.horizon,
                'n_agents': Config.n_agents,
                'state_dim': Config.state_dim,
                'act_dim': Config.act_dim,
                'max_path_length': Config.max_path_length,
                'config': safe_config,
            }

            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            
            if has_kwargs:
                model_args = {k: v for k, v in potential_args.items() if v is not None}
            else:
                model_args = {k: v for k, v in potential_args.items() if k in sig.parameters and v is not None}
            
            model_config = utils.Config(
                model_class_path,
                **model_args
            )
            model = model_config()
            print(f"[INFO] 模型实例化成功: {type(model)}")
        except Exception as e:
            print(f"[ERROR] 模型实例化失败: {e}")
            raise e

        print("[DEBUG] _init: 配置 Trainer...")
        trainer_config = utils.Config(
            'diffuser.utils.madc_training.MADCTrainer',
            model=model,
            dataset=dataset,
            learning_rate=Config.learning_rate,
        )
        self.trainer = trainer_config()

        print("[DEBUG] _init: 加载环境...")
        try:
            env_mod_name = {
                "mamujoco": "diffuser.datasets.mamujoco",
            }[Config.env_type]
            env_mod = __import__(env_mod_name, fromlist=[None])
            self.env_list = [env_mod.load_environment(Config.dataset) for _ in range(Config.num_eval)]
            print(f"[INFO] 成功启动 {len(self.env_list)} 个评估环境")
        except Exception as e:
            print(f"[ERROR] 环境加载失败: {e}")
            raise e
        
        self.initialized = True
        print("[DEBUG] _init: 初始化完成，发送信号...")
        self.p.send("init_done")

    def run(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                torch.tensor([0.0], device='cuda') 
            except Exception as e:
                print(f"[WARN] 子进程 CUDA 初始化尝试失败: {e}")

        self.parent_remote.close()
        if not self.verbose:
            sys.stdout = open(os.devnull, "w")
        try:
            while True:
                try:
                    cmd, data = self.queue.get()
                except EOFError:
                    self.p.close(); break
                if cmd == "init": self._init(**data)
                elif cmd == "evaluate": self._evaluate(**data)
                elif cmd == "close":
                    self.p.send("closed"); self.p.close(); break
                else:
                    self.p.close(); raise NotImplementedError(f"Unknown command {cmd}")
                time.sleep(1)
        except KeyboardInterrupt:
            self.p.close()
        except Exception as e:
            logger.log(f"Worker process crashed with error: {e}")
            raise e
        finally:
            self.p.close()


class MADCEvaluator:
    def __init__(self, **kwargs):
        multiprocessing.set_start_method("spawn", force=True)
        self.parent_remote, self.child_remote = Pipe()
        self.queue = multiprocessing.Queue()
        self._worker_process = MADCEvaluatorWorker(
            parent_remote=self.parent_remote,
            child_remote=self.child_remote,
            queue=self.queue,
            **kwargs,
        )
        self._worker_process.start()
        self.child_remote.close()

    def init(self, **kwargs):
        self.queue.put(["init", kwargs])

    def evaluate(self, **kwargs):
        self.queue.put(["evaluate", kwargs])

    def __del__(self):
        try:
            self.queue.put(["close", None])
            self.parent_remote.recv()
            self._worker_process.join()
        except (BrokenPipeError, EOFError, AttributeError, FileNotFoundError):
            pass
        if self._worker_process.is_alive():
            self._worker_process.terminate()