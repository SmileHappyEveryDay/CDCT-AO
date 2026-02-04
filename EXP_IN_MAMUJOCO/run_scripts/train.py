import argparse
import os

import diffuser.utils as utils
import torch
import yaml
from diffuser.utils.launcher_util import (
    build_config_from_dict,
    discover_latest_checkpoint_path,
)

# 导入 MADC 所需的模块
from diffuser.models.madc import MADC_BC, GPTConfig
from diffuser.utils.madc_training import MADCTrainer


def main(Config, RUN):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    utils.set_seed(Config.seed)
    dataset_extra_kwargs = dict()

    # --- 默认配置部分 ---
    Config.discrete_action = getattr(Config, "discrete_action", False)
    Config.state_loss_weight = getattr(Config, "state_loss_weight", None)
    Config.opponent_loss_weight = getattr(Config, "opponent_loss_weight", None)
    Config.use_seed_dataset = getattr(Config, "use_seed_dataset", False)
    Config.residual_attn = getattr(Config, "residual_attn", True)
    Config.use_temporal_attention = getattr(Config, "use_temporal_attention", True)
    Config.env_ts_condition = getattr(Config, "env_ts_condition", False)
    Config.use_return_to_go = getattr(Config, "use_return_to_go", False)
    Config.joint_inv = getattr(Config, "joint_inv", False)
    Config.use_zero_padding = getattr(Config, "use_zero_padding", True)
    Config.use_inv_dyn = getattr(Config, "use_inv_dyn", True)
    Config.pred_future_padding = getattr(Config, "pred_future_padding", False)
    Config.decentralized_execution = getattr(Config, "decentralized_execution", False)
    if not hasattr(Config, "agent_condition_type"):
        if Config.decentralized_execution:
            Config.agent_condition_type = "single"
        else:
            Config.agent_condition_type = "all"
    Config.n_head = getattr(Config, "n_head", 8)
    Config.n_layer = getattr(Config, "n_layer", 6)
    Config.n_embd = getattr(Config, "n_embd", 128)
    Config.model_type = getattr(Config, "model_type", "rtgs_state_action")

    # --- 数据集加载 ---
    dataset_config = utils.Config(
        Config.loader,
        savepath="dataset_config.pkl",
        env_type=Config.env_type,
        env=Config.dataset,
        n_agents=Config.n_agents,
        horizon=Config.horizon,
        history_horizon=Config.history_horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        max_n_episodes=Config.max_n_episodes,
        use_padding=Config.use_padding,
        use_action=Config.use_action,
        discrete_action=Config.discrete_action,
        max_path_length=Config.max_path_length,
        # [!! 修改 !!] 强制关闭 returns
        include_returns=False, 
        include_env_ts=True,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
        termination_penalty=Config.termination_penalty,
        agent_share_parameters=True,
        use_seed_dataset=Config.use_seed_dataset,
        seed=Config.seed,
        use_inv_dyn=Config.use_inv_dyn,
        decentralized_execution=Config.decentralized_execution,
        use_zero_padding=Config.use_zero_padding,
        agent_condition_type=Config.agent_condition_type,
        pred_future_padding=Config.pred_future_padding,
        **dataset_extra_kwargs,
    )
    dataset = dataset_config()
    
    # [!! 新增 !!] 从 dataset 获取维度信息
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    render_config = utils.Config(
        utils.MAMujocoRenderer, # 现在 utils.MAMujocoRenderer 应该能正常工作了
        savepath="render_config.pkl",
        env=Config.dataset,
        env_type=Config.env_type, # [!! 新增 !!] 添加缺失的 env_type 参数
    )
    renderer = render_config()

    # --- MADC 模型与训练器配置 ---
    gpt_config = GPTConfig(
        state_dim=observation_dim, # 现在 observation_dim 已经定义了
        act_dim=action_dim,        # 确保 action_dim 也被定义
        block_size=Config.horizon,
        n_layer=Config.n_layer,
        n_head=Config.n_head,
        n_embd=Config.n_embd,
        model_type=Config.model_type,
        max_timestep=Config.max_path_length,
    )
    model = MADC_BC(gpt_config)

    trainer_config = utils.Config(
        utils.MADCTrainer,
        savepath="trainer_config.pkl",
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        log_freq=Config.log_freq,
        save_freq=Config.save_freq,
        eval_freq=Config.eval_freq,
        bucket=logger.root,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
    )

    # 现在 renderer 已经定义了，可以传入
    trainer = trainer_config(model, dataset, renderer) 

    # --- Evaluator 和继续训练的逻辑 ---
    if Config.eval_freq > 0:
        renderer = None
        if Config.eval_render:
            render_config = utils.Config(Config.renderer, savepath="render_config.pkl", env_type=Config.env_type, env=Config.dataset)
            renderer = render_config()
        evaluator_config = utils.Config(Config.evaluator, savepath="evaluator_config.pkl", verbose=False)
        evaluator = evaluator_config()
        evaluator.init(log_dir=logger.prefix, renderer=renderer)
        trainer.set_evaluator(evaluator)
        
    if Config.continue_training:
        loadpath = discover_latest_checkpoint_path(os.path.join(trainer.bucket, logger.prefix, "checkpoint"))
        if loadpath:
            trainer.load(loadpath)

    # --- 报告参数并开始训练 ---
    utils.report_parameters(model)
    logger.print("✓ All systems go. Starting training...")

    # [!! 新增 !!] 显式保存初始模型 (state_0.pt)
    # 这会创建 checkpoint 文件夹并保存初始权重
    if trainer.step == 0 and Config.save_checkpoints:
        trainer.save(0)

    n_epochs = int((Config.n_train_steps - trainer.step) // Config.n_steps_per_epoch)

    for i in range(n_epochs):
        logger.print(f"Epoch {i} / {n_epochs} | {logger.prefix}")
        trainer.train(n_train_steps=Config.n_steps_per_epoch)
    trainer.finish_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=str, default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)

    from ml_logger import RUN, logger

    Config = build_config_from_dict(exp_specs)
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    job_name = Config.job_name.format(**vars(Config))
    RUN.prefix, RUN.job_name, _ = RUN(
        script_path=__file__,
        exp_name=exp_specs["exp_name"],
        job_name=job_name + f"/{Config.seed}",
    )

    logger.configure(RUN.prefix, root=RUN.script_root)
    logger.remove("traceback.err")
    logger.remove("parameters.pkl")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text(
        """
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: mse_loss
                      xKey: steps
                    """,
        filename=".charts.yml",
        dedent=True,
        overwrite=True,
    )
    logger.save_yaml(exp_specs, "exp_specs.yml")

    main(Config, RUN)