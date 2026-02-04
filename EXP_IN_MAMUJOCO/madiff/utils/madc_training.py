import os
import torch
import numpy as np
from ml_logger import logger
from .timer import Timer

# [!! 修复 1 !!] 递归处理字典的 to_device
def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [to_device(v, device) for v in x]
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    else:
        return x

# [!! 修复 2 !!] 正确处理嵌套字典 cond 的 collate_fn
def madc_collate_fn(batch):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        if key == "cond":
            # cond 是一个字典 {'x': ..., 'masks': ...}
            cond_keys = batch[0][key].keys()
            collated_batch[key] = {
                k: torch.from_numpy(np.stack([d[key][k] for d in batch]))
                for k in cond_keys
            }
        else:
            # 对于普通字段 (x, loss_masks 等)，直接堆叠
            list_of_arrays = [d[key] for d in batch]
            collated_batch[key] = torch.from_numpy(np.stack(list_of_arrays))
            
    return collated_batch

def cycle(dl):
    while True:
        for data in dl:
            yield data

class MADCTrainer(object):
    def __init__(
        self,
        model,
        dataset,
        renderer=None, # [!! 修改 !!] 设置默认值为 None
        train_batch_size=32,
        train_lr=1e-4,
        train_device="cpu",
        log_freq=100,
        save_freq=1000,
        eval_freq=1000,
        bucket=None,
        save_checkpoints=True,
        # [!! 修复 !!] 添加 **kwargs 以兼容多余参数 (如 learning_rate)
        **kwargs 
    ):
        self.model = model
        self.dataset = dataset
        self.renderer = renderer
        self.train_batch_size = train_batch_size
        
        # [!! 修复 !!] 优先使用 kwargs 中的 learning_rate (如果存在)
        if 'learning_rate' in kwargs:
            train_lr = kwargs['learning_rate']
            
        self.train_lr = train_lr
        self.train_device = train_device
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.bucket = bucket
        self.save_checkpoints = save_checkpoints

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_lr)
        
        self.dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=train_batch_size,
                num_workers=2,
                shuffle=True,
                pin_memory=True,
                collate_fn=madc_collate_fn,
            )
        )
        self.device = train_device
        
        # [!! 修复 !!] 初始化全局 step 计数器
        self.step = 0
        self.evaluator = None

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def train(self, n_train_steps):
        self.model.to(self.device)
        self.model.train()

        timer = Timer()
        for i in range(n_train_steps):
            # [!! 修复 !!] 增加全局 step
            self.step += 1
            
            batch = next(self.dataloader)
            batch = to_device(batch, self.device)

            loss, infos = self.model.loss(**batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # [!! 修复 !!] 使用 self.step 而不是局部变量 i
            if self.step % self.log_freq == 0:
                print(f"Step {self.step}: Loss = {loss.item():.4f}")
                logger.log_metrics(infos, step=self.step)

            if self.save_checkpoints and self.step % self.save_freq == 0:
                self.save(self.step)
            
            # 如果配置了评估器，执行评估
            if self.evaluator and self.step % self.eval_freq == 0:
                self.evaluator.evaluate(load_step=self.step)
    
    def save(self, step):
        if self.bucket is None:
            return
        
        data = {"step": step, "model": self.model.state_dict()}
        savepath = os.path.join(self.bucket, logger.prefix, "checkpoint")
        os.makedirs(savepath, exist_ok=True)
        
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f"state_{step}.pt")
        else:
            savepath = os.path.join(savepath, "state.pt")
            
        torch.save(data, savepath)
        logger.print(f"Saved model to {savepath}")

    def load(self, loadpath):
        data = torch.load(loadpath, map_location=self.device)
        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        logger.print(f"Loaded model from {loadpath} at step {self.step}")
    
    def finish_training(self):
        self.save(self.step)