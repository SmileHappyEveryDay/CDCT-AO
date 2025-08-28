"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import copy
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

from .utils import MultiStageAdaptiveLRScheduler

class TrainerConfig:
    # optimization parameters
    max_epochs = 1000
    batch_size = 128
    learning_rate = 5e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 0.5
    weight_decay = 0.1  # only applied on matmul weights
    # checkpoint settings
    num_workers = 0  # for DataLoader
    use_lr_scheduler = False 
    use_curiosity = True           
    feature_dim = 128              
    curiosity_lr = 1e-4            
    forward_loss_coef = 0.2        
    inverse_loss_coef = 0.8        
    intrinsic_reward_coef = 0.01  
    curiosity_decay = 0.99         

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
class Trainer:

    def __init__(self, model, critic_model, config):
        self.model = model
        self.critic_model = critic_model
        self.config = config

        
        # self.current_rtg = config.rtg_min
        # self.running_win_rate = 0.0
        # self.running_return = 0.0
    
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = self.raw_model.configure_optimizers(config, config.learning_rate)

        self.raw_critic_model = self.critic_model.module if hasattr(self.critic_model, "module") else self.critic_model
        self.critic_optimizer = self.raw_critic_model.configure_optimizers(config, config.learning_rate * 10)
         
        #add lr scheduler
        if self.config.use_lr_scheduler: 
            self.scheduler = MultiStageAdaptiveLRScheduler(
                self.optimizer,
                win_rate_threshold=0.6,
                win_rate_window=50,
                lr_bounds=(1e-5, 5e-5),
                warmup_epochs=5,
                exploration_epochs=20,
                restart_period=100,
                restart_factor=1.5
        )
        
            self.critic_scheduler = MultiStageAdaptiveLRScheduler(
                self.critic_optimizer,
                win_rate_threshold=0.6,
                win_rate_window=50,
                lr_bounds=(1e-4, 5e-4), 
                warmup_epochs=5,
                exploration_epochs=20,
                restart_period=100,
                restart_factor=1.5
        )

        if hasattr(config, 'use_curiosity') and config.use_curiosity:
            from models.curiosity_model import IntrinsicCuriosityModule
            self.curiosity_module = IntrinsicCuriosityModule(config)
            if torch.cuda.is_available():
                self.curiosity_module = torch.nn.DataParallel(self.curiosity_module).to(self.device)
            self.curiosity_optimizer = torch.optim.Adam(
                self.curiosity_module.parameters(), 
                lr=config.curiosity_lr
            )
    def train(self, dataset, train_critic=True):
        model, critic_model, config = self.raw_model, self.raw_critic_model, self.config
        target_model = copy.deepcopy(model)
        target_model.train(False)
        
        def run_epoch():
            model.train(True)
            critic_model.train(True)
            if self.config.mode == "offline":
                loader = DataLoader(dataset, shuffle=True, pin_memory=True, drop_last=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)
            elif self.config.mode == "online":
                loader = DataLoader(dataset, shuffle=True, pin_memory=True, drop_last=True,
                                    batch_size=dataset.__len__(),
                                    num_workers=config.num_workers)
            else:
                raise NotImplementedError

            loss_info = 0
            kl_loss_info = 0
            pbar = tqdm(enumerate(loader), total=len(loader))

            # todo: check these inputs
            for it, (s, o, a, r, ava, v, rtg, ret, adv, t, pre_a, next_s, next_rtg, done) in pbar:
                s = s.to(self.device)
                o = o.to(self.device)
                a = a.to(self.device)
                r = r.to(self.device)
                ava = ava.to(self.device)
                v = v.to(self.device)
                rtg = rtg.to(self.device)
                ret = ret.to(self.device)
                adv = adv.to(self.device)
                t = t.to(self.device)
                pre_a = pre_a.to(self.device)
                next_s = next_s.to(self.device)
                next_rtg = next_rtg.to(self.device)
                done = done.to(self.device)
                curiosity_loss = torch.tensor(0.0, device=self.device)
                intrinsic_rewards = torch.zeros_like(r)
                if hasattr(self, 'curiosity_module') and hasattr(config, 'use_curiosity') and config.use_curiosity:
                    try:
                        
                        if next_s is not None and next_s.shape[1] >= 1:                     
                            batch_size, seq_len, obs_dim = o.shape
                            _, _, next_obs_dim = next_s.shape 
                            current_obs = o.reshape(-1, obs_dim)  
                            next_obs = next_s.reshape(-1, next_obs_dim)  
                            actions = a.reshape(-1)  
                            valid_mask = (actions >= 0) & (actions < config.action_dim)
                            if valid_mask.sum() > 0:
                                current_obs_valid = current_obs[valid_mask]
                                next_obs_valid = next_obs[valid_mask]
                                actions_valid = actions[valid_mask]
                                intrinsic_reward, pred_action, pred_next_feat, next_feat = \
                                    self.curiosity_module(current_obs_valid, next_obs_valid, actions_valid)
                                forward_loss = F.mse_loss(pred_next_feat, next_feat.detach())
                                inverse_loss = F.cross_entropy(pred_action, actions_valid)
                                curiosity_loss = config.forward_loss_coef * forward_loss + \
                                               config.inverse_loss_coef * inverse_loss
                                full_intrinsic_reward = torch.zeros_like(actions, dtype=torch.float32, device=self.device)
                                full_intrinsic_reward[valid_mask] = intrinsic_reward.squeeze(-1)
                                full_intrinsic_reward = full_intrinsic_reward.view(batch_size, seq_len, 1)
                                intrinsic_rewards = full_intrinsic_reward * config.intrinsic_reward_coef    
                            else:
                                print("✗ no valid actions")
                                pass
                        elif o.shape[1] > 1:  
                            batch_size, seq_len, obs_dim = o.shape
                            current_obs = o[:, :-1].contiguous().view(-1, obs_dim)
                            next_obs = o[:, 1:].contiguous().view(-1, obs_dim)
                            actions = a[:, :-1].contiguous().view(-1)
                            valid_mask = (actions >= 0) & (actions < config.action_dim)
                            if valid_mask.sum() > 0:
                                current_obs_valid = current_obs[valid_mask]
                                next_obs_valid = next_obs[valid_mask]
                                actions_valid = actions[valid_mask]
                                intrinsic_reward, pred_action, pred_next_feat, next_feat = \
                                    self.curiosity_module(current_obs_valid, next_obs_valid, actions_valid)
                                forward_loss = F.mse_loss(pred_next_feat, next_feat.detach())
                                inverse_loss = F.cross_entropy(pred_action, actions_valid)
                                curiosity_loss = config.forward_loss_coef * forward_loss + \
                                               config.inverse_loss_coef * inverse_loss
                                full_intrinsic_reward = torch.zeros_like(actions, dtype=torch.float32, device=self.device)
                                full_intrinsic_reward[valid_mask] = intrinsic_reward.squeeze(-1)
                                full_intrinsic_reward = full_intrinsic_reward.view(batch_size, seq_len-1, 1)
                                intrinsic_rewards[:, :-1] = full_intrinsic_reward * config.intrinsic_reward_coef
                               
                            else:
                                print("✗ no valid actions")
                        else:
                            print(f"✗ : seq_len={o.shape[1]}")
                        
                    except Exception as e:
                        print(f"✗ wrong: {e}")
                        import traceback
                        traceback.print_exc()
                        curiosity_loss = torch.tensor(0.0, device=self.device)
                else:
                    if config.mode == "online": 
                        print("✗ curiosity module not enabled or not exist")
                intrinsic_rewards = torch.clamp(intrinsic_rewards, -0.001, 0.001)
                enhanced_adv = adv + intrinsic_rewards.detach()

                with torch.set_grad_enabled(True):
                    logits = model(o, pre_a, rtg, t)

                    if self.config.mode == "offline":
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), a.reshape(-1))
                        entropy_info = 0.
                        ratio_info = 0.
                        confidence_info = 0.
                    elif self.config.mode == "online":
                        enhanced_adv_flat = enhanced_adv.reshape(-1, enhanced_adv.size(-1))

                        logits[ava == 0] = -1e10
                        distri = Categorical(logits=logits.reshape(-1, logits.size(-1)))
                        target_a = a.reshape(-1)
                        log_a = distri.log_prob(target_a).unsqueeze(-1)

                        old_logits = target_model(o, pre_a, rtg, t).detach()
                        old_logits[ava == 0] = -1e10
                        old_distri = Categorical(logits=old_logits.reshape(-1, old_logits.size(-1)))
                        old_log_a = old_distri.log_prob(target_a).unsqueeze(-1)

                        imp_weights = torch.exp(log_a - old_log_a)
                        actor_loss_ori = imp_weights * enhanced_adv_flat
                        actor_loss_clip = torch.clamp(imp_weights, 1.0 - 0.2, 1.0 + 0.2) * enhanced_adv_flat
                        actor_loss = -torch.min(actor_loss_ori, actor_loss_clip)

                        act_entropy = distri.entropy().unsqueeze(-1)
                        loss = actor_loss - 0.01 * act_entropy

                        entropy_info = act_entropy.mean().item()
                        ratio_info = imp_weights.mean().item()
                        confidence_info = torch.exp(log_a).mean().item()
                    else:
                        raise NotImplementedError
                    
                    loss = loss.mean()
                    loss_info = loss.item()

                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                critic_loss_info = 0.
                if train_critic:
                    with torch.set_grad_enabled(True):
                        v_value = critic_model(s, pre_a, rtg, t)
                        v_clip = v + (v_value - v).clamp(-0.2, 0.2)
                        critic_loss_ori = F.smooth_l1_loss(v_value.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss_clip = F.smooth_l1_loss(v_clip.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss = torch.max(critic_loss_ori, critic_loss_clip)

                        critic_loss_info = critic_loss.mean().item()

                    critic_model.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_model.parameters(), config.grad_norm_clip)
                    self.critic_optimizer.step()
                if hasattr(self, 'curiosity_module') and curiosity_loss.item() > 0:
                    self.curiosity_optimizer.zero_grad()
                    curiosity_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.curiosity_module.parameters(), 
                        config.grad_norm_clip
                    )
                    self.curiosity_optimizer.step()
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}, curiosity loss {curiosity_loss.item():.5f}")
            
            return loss_info, critic_loss_info, entropy_info, ratio_info, confidence_info

        actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = 0., 0., 0., 0., 0.
        for epoch in range(config.max_epochs):
            actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = run_epoch()
        return actor_loss_ret, critic_loss_ret, entropy, ratio, confidence
    def update_scheduler(self, win_rate, sample_return):

        if not hasattr(self, 'scheduler'):
            return
        try:
            self.scheduler.update_win_rate(win_rate)
            self.scheduler.step()
            if hasattr(self, 'critic_scheduler'):
                self.critic_scheduler.step()
            print(f"\n=== update lr ===")
            print(f"win rate: {win_rate:.3f}")
            print(f"return: {sample_return:.3f}")
            print(f"Actor lr: {self.optimizer.param_groups[0]['lr']:.6f}")
            if hasattr(self, 'critic_optimizer'):
                print(f"Critic lr: {self.critic_optimizer.param_groups[0]['lr']:.6f}")
            print("===============\n")
        except Exception as e:
            print(f"✗ wrong with update_scheduler: {str(e)}")