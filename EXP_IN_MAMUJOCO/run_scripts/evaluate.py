import argparse
import os
import pickle
import yaml
import time
import torch # [!! 新增 !!] 导入 torch 以检测 device

import diffuser.utils as utils

class Parser(utils.Parser):
    dataset: str = '2ant-Good'
    config: str = 'config.locomotion'
    num_eval: int = 10
    # [!! 新增 !!] 确保定义了 device 参数
    device: str = 'cuda' 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.experiment, "r") as spec_file:
        eval_config = yaml.safe_load(spec_file)

    log_dir = eval_config['log_dir']
    print(f"评估日志目录: {log_dir}")

    evaluator_path = eval_config['evaluator']
    print(f"使用评估器: {evaluator_path}")
    evaluator_config = utils.Config(evaluator_path, verbose=True)
    evaluator = evaluator_config()

    # [!! 关键修复 !!]
    # 使用同步等待机制
    
    print("\n--- 正在初始化评估器... ---")
    
    # [!! 修复 !!] 准备参数，不再依赖未定义的 Config 对象
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_eval = eval_config.get('num_eval', 10)

    evaluator.init(
        log_dir=log_dir,
        # [!! 修复 !!] 传入本地变量
        device=device, 
        num_eval=num_eval
    )
    # 阻塞并等待，直到收到 "init_done" 信号
    evaluator.parent_remote.recv()
    print("--- 评估器初始化完成。 ---")

    evaluation_performed = False
    
    load_steps = eval_config.get('load_steps', [])
    if not load_steps:
        # 如果 load_steps 为空或不存在，则创建一个特殊列表来评估 state.pt
        load_steps = [None] 

    for load_step in load_steps:
        step_id = "latest" if load_step is None else load_step
        ckpt_name = "state.pt" if load_step is None else f"state_{load_step}.pt"
        
        print(f"\n--- 准备评估 Checkpoint: {step_id} ({ckpt_name}) ---")
        
        ckpt_path = os.path.join(log_dir, "checkpoint", ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"    [跳过] Checkpoint 文件 {ckpt_path} 不存在。")
            continue

        results_path = os.path.join(log_dir, f"results/step_{step_id}-ep_{eval_config['num_eval']}.json")
        if not eval_config.get('overwrite', False) and os.path.exists(results_path):
            print(f"    [跳过] 结果文件 {results_path} 已存在，且 overwrite=False。")
            continue
        
        print(f"    [执行] 开始评估...")
        evaluator.evaluate(load_step=load_step)
        
        # 阻塞并等待，直到收到 "evaluation_done" 信号
        evaluator.parent_remote.recv()
        print(f"    [完成] Checkpoint {step_id} 评估完成。")
        evaluation_performed = True

    if not evaluation_performed:
        print("\n--- 没有执行任何有效的评估任务。 ---")

    print("\n--- 评估流程结束，正在关闭评估器... ---")
    del evaluator
    time.sleep(2) # 短暂等待以确保资源释放
    print("--- 评估器已关闭。 ---")