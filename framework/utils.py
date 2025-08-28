import copy
import random
import numpy as np
import torch
from torch.nn import functional as F
from gym.spaces.discrete import Discrete
# from sc2.toy_data import toy_example

from torch.optim.lr_scheduler import _LRScheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample(model, critic_model, state, obs, sample=False, actions=None, rtgs=None,
           timesteps=None, available_actions=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    if torch.cuda.is_available():
        block_size = model.module.get_block_size()
    else:
        block_size = model.get_block_size()
    model.eval()
    critic_model.eval()

    # x: (batch_size, context_length, dim)
    # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
    obs_cond = obs if obs.size(1) <= block_size//3 else obs[:, -block_size//3:] # crop context if needed
    state_cond = state if state.size(1) <= block_size//3 else state[:, -block_size//3:] # crop context if needed
    if actions is not None:
        actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
    rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed
    timesteps = timesteps if timesteps.size(1) <= block_size//3 else timesteps[:, -block_size//3:] # crop context if needed

    logits = model(obs_cond, pre_actions=actions, rtgs=rtgs, timesteps=timesteps)
    # pluck the logits at the final step and scale by temperature
    logits = logits[:, -1, :]
    # apply softmax to convert to probabilities
    if available_actions is not None:
        logits[available_actions == 0] = -1e10
    probs = F.softmax(logits, dim=-1)

    if sample:
        a = torch.multinomial(probs, num_samples=1)
    else:
        _, a = torch.topk(probs, k=1, dim=-1)

    v = critic_model(state_cond, pre_actions=actions, rtgs=rtgs, timesteps=timesteps).detach()
    v = v[:, -1, :]

    return a, v



def get_dim_from_space(space):
    if isinstance(space[0], Discrete):
        return space[0].n
    elif isinstance(space[0], list):
        return space[0][0]


def padding_obs(obs, target_dim):
    len_obs = np.shape(obs)[-1]
    if len_obs > target_dim:
        print("target_dim (%s) too small, obs dim is %s." % (target_dim, len(obs)))
        raise NotImplementedError
    elif len_obs < target_dim:
        padding_size = target_dim - len_obs
        if isinstance(obs, list):
            obs = np.array(copy.deepcopy(obs))
            padding = np.zeros(padding_size)
            obs = np.concatenate((obs, padding), axis=-1).tolist()
        elif isinstance(obs, np.ndarray):
            obs = copy.deepcopy(obs)
            shape = np.shape(obs)
            padding = np.zeros((shape[0], shape[1], padding_size))
            obs = np.concatenate((obs, padding), axis=-1)
        else:
            print("unknwon type %s." % type(obs))
            raise NotImplementedError
    return obs


def padding_ava(ava, target_dim):
    len_ava = np.shape(ava)[-1]
    if len_ava > target_dim:
        print("target_dim (%s) too small, ava dim is %s." % (target_dim, len(ava)))
        raise NotImplementedError
    elif len_ava < target_dim:
        padding_size = target_dim - len_ava
        if isinstance(ava, list):
            ava = np.array(copy.deepcopy(ava), dtype=np.long)
            padding = np.zeros(padding_size, dtype=np.long)
            ava = np.concatenate((ava, padding), axis=-1).tolist()
        elif isinstance(ava, np.ndarray):
            ava = copy.deepcopy(ava)
            shape = np.shape(ava)
            padding = np.zeros((shape[0], shape[1], padding_size), dtype=np.long)
            ava = np.concatenate((ava, padding), axis=-1)
        else:
            print("unknwon type %s." % type(ava))
            raise NotImplementedError
    return ava




class MultiStageAdaptiveLRScheduler(_LRScheduler):
    def __init__(self, optimizer, 
                 win_rate_threshold=0.6,
                 win_rate_window=100, 
                 lr_bounds=(1e-5, 1e-3),
                 momentum=0.9,
                 warmup_epochs=10,
                 exploration_epochs=50,
                 restart_period=200,
                 restart_factor=2.0):
      
        self.optimizer = optimizer
        self.win_rate_threshold = win_rate_threshold
        self.win_rate_window = win_rate_window
        self.lr_min, self.lr_max = lr_bounds
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.exploration_epochs = exploration_epochs
        self.restart_period = restart_period
        self.restart_factor = restart_factor
        self.win_history = []
        self.current_win_rate = 0.0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        self.last_restart_epoch = 0
        self.stage = 'warmup'
        
        super().__init__(optimizer)
        
    def update_win_rate(self, won):
        
        self.win_history.append(1 if won else 0)
        if len(self.win_history) > self.win_rate_window:
            self.win_history.pop(0)
        self.current_win_rate = np.mean(self.win_history)
        
    def _determine_stage(self):
        
        if self.current_epoch < self.warmup_epochs:
            return 'warmup'
        elif self.current_epoch < self.exploration_epochs:
            return 'exploration'
        elif self.current_win_rate > 0.8:
            return 'fine_tuning'
        else:
            return 'normal'
        
    def _check_restart(self):
        
        epochs_since_restart = self.current_epoch - self.last_restart_epoch
        if epochs_since_restart >= self.restart_period:
            self.last_restart_epoch = self.current_epoch
            return True
        return False
        
    def get_lr_multiplier(self): 
        if self._check_restart():
            return self.restart_factor
        stage = self._determine_stage()
        
        if stage == 'warmup':        
            return (self.current_epoch + 1) / self.warmup_epochs          
        elif stage == 'exploration':      
            return 1.2     
        elif stage == 'fine_tuning':       
            return 0.5   
        else:  # normal stage
            if len(self.win_history) < self.win_rate_window // 2:
                return 1.0  
            win_rate_diff = self.win_rate_threshold - self.current_win_rate
            if win_rate_diff > 0:             
                multiplier = 1.0 + np.tanh(win_rate_diff * 2) * 0.2
            else:          
                multiplier = 1.0 - np.tanh(-win_rate_diff * 2) * 0.1           
            multiplier = self.momentum * 1.0 + (1 - self.momentum) * multiplier
            return multiplier
            
    def step(self, metrics=None, epoch=None):
        self.current_epoch = epoch if epoch is not None else self.current_epoch + 1
        multiplier = self.get_lr_multiplier()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            new_lr = base_lr * multiplier 
            new_lr = np.clip(new_lr, self.lr_min, self.lr_max)
            param_group['lr'] = new_lr
            
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    def state_dict(self):
        
        return {
            'base_lrs': self.base_lrs,
            'current_epoch': self.current_epoch,
            'win_history': self.win_history,
            'current_win_rate': self.current_win_rate,
            'last_restart_epoch': self.last_restart_epoch,
            'stage': self.stage
        }
    
    def load_state_dict(self, state_dict):
        self.base_lrs = state_dict['base_lrs']
        self.current_epoch = state_dict['current_epoch']
        self.win_history = state_dict['win_history']
        self.current_win_rate = state_dict['current_win_rate']
        self.last_restart_epoch = state_dict['last_restart_epoch']
        self.stage = state_dict['stage']

