import setproctitle
import logging
import argparse
import torch
import sys
import os

from tensorboardX.writer import SummaryWriter
from framework.utils import set_seed
from framework.trainer import Trainer, TrainerConfig
from framework.utils import get_dim_from_space
from envs.env import Env
from framework.buffer import ReplayBuffer
from framework.rollout import RolloutWorker
from datetime import datetime, timedelta
from models.gpt_model import GPT, GPTConfig
from envs import config

# args = sys.argv[1:]
parser = argparse.ArgumentParser()

parser.add_argument('--cuda_id', type=str, default='0',help='use GPU ID')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=1)
parser.add_argument('--model_type', type=str, default='rtgs_state_action')
parser.add_argument('--eval_episodes', type=int, default=64)#32
parser.add_argument('--max_timestep', type=int, default=400)
parser.add_argument('--log_dir', type=str, default='./logs/')
parser.add_argument('--save_log', type=bool, default=True)
parser.add_argument('--exp_name', type=str, default='easy_trans')
parser.add_argument('--pre_train_model_path', type=str, default='../offline_model/')

parser.add_argument('--offline_map_lists', type=list, default=['MMM2','MMM2','MMM2','MMM2','MMM2'])
parser.add_argument('--offline_episode_num', type=list, default=[200, 200, 200, 200, 200])
parser.add_argument('--offline_data_quality', type=list, default=['good'])
parser.add_argument('--offline_data_dir', type=str, default='../../PATH/TO/YOUR_DATA')

parser.add_argument('--offline_epochs', type=int, default=10)
parser.add_argument('--offline_mini_batch_size', type=int, default=128)
parser.add_argument('--offline_lr', type=float, default=1e-4)
parser.add_argument('--offline_eval_interval', type=int, default=1)
parser.add_argument('--offline_train_critic', type=bool, default=True)
parser.add_argument('--offline_model_save', type=bool, default=True)

parser.add_argument('--online_buffer_size', type=int, default=64)
parser.add_argument('--online_epochs', type=int, default=5000)
parser.add_argument('--online_ppo_epochs', type=int, default=15)
parser.add_argument('--online_lr', type=float, default=5e-4)
parser.add_argument('--online_eval_interval', type=int, default=1)
parser.add_argument('--online_train_critic', type=bool, default=True)
parser.add_argument('--online_pre_train_model_load', type=bool, default=False)
parser.add_argument('--online_pre_train_model_id', type=int, default=9)
parser.add_argument('--use_curiosity', action='store_true', default=True, 
                   help='curiosity-driven exploration')
parser.add_argument('--feature_dim', type=int, default=128, 
                   help='Dimension of the feature space for the curiosity module')
parser.add_argument('--curiosity_lr', type=float, default=5e-4, 
                   help='Learning rate for the curiosity module')
parser.add_argument('--intrinsic_reward_coef', type=float, default=0.01, 
                   help='Intrinsic reward coefficient')
parser.add_argument('--forward_loss_coef', type=float, default=0.02, 
                   help='Forward model loss coefficient')
parser.add_argument('--inverse_loss_coef', type=float, default=0.1, 
                   help='Inverse model loss coefficient')


def get_env_dims(env):
    """get dimensions from environment"""
    try:
        global_obs_dim = get_dim_from_space(env.real_env.share_observation_space)
        local_obs_dim = get_dim_from_space(env.real_env.observation_space)
        action_dim = get_dim_from_space(env.real_env.action_space)

        print(f"get dimensions: global={global_obs_dim}, local={local_obs_dim}, action={action_dim}")
        return global_obs_dim, local_obs_dim, action_dim
    except Exception as e:
        print(f"get dimensions false : {e}")
        return None
def setup_device(args):
    """set up training device"""
    if not args.use_cpu and torch.cuda.is_available():
        # set visible CUDA devices
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
        device = torch.device("cuda")
        print(f"using GPU: {args.cuda_id}")
        print(f"current GPU model: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("using CPU for training")
    return device

def setup_process_title(args):
    """set process title"""
    title = f"{args.exp_name}"
    setproctitle.setproctitle(title)
    print(f"process title set to: {title}")
# args = parser.parse_args(args, parser)
args = parser.parse_args()
set_seed(args.seed)
setup_process_title(args)
torch.set_num_threads(8)
cur_time = datetime.now() + timedelta(hours=0)
args.log_dir += cur_time.strftime("[%m-%d]%H.%M.%S")
writter = SummaryWriter(args.log_dir) if args.save_log else None
eval_env = Env(args.eval_episodes)
online_train_env = Env(args.online_buffer_size)
dims = get_env_dims(online_train_env)

if dims is None:
    print("using default dimensions")
    global_obs_dim = 299
    local_obs_dim = 252
    action_dim = 15
else:
    global_obs_dim, local_obs_dim, action_dim = dims
block_size = args.context_length * 3
print("global_obs_dim: ", global_obs_dim)
print("local_obs_dim: ", local_obs_dim)
print("action_dim: ", action_dim)
mconf_actor = GPTConfig(local_obs_dim, action_dim, block_size,
                        n_layer=2, n_head=2, n_embd=32, model_type=args.model_type, max_timestep=args.max_timestep)
model = GPT(mconf_actor, model_type='actor')
mconf_critic = GPTConfig(global_obs_dim, action_dim, block_size,
                         n_layer=2, n_head=2, n_embd=32, model_type=args.model_type, max_timestep=args.max_timestep)
critic_model = GPT(mconf_critic, model_type='critic')
# device = setup_device(args)
# model = model.to(device)
# critic_model = critic_model.to(device)

if torch.cuda.is_available():  #torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = torch.nn.DataParallel(model).to(device)
    critic_model = torch.nn.DataParallel(critic_model).to(device)
buffer = ReplayBuffer(block_size, global_obs_dim, local_obs_dim, action_dim)
rollout_worker = RolloutWorker(model, critic_model, buffer, global_obs_dim, local_obs_dim, action_dim)
used_data_dir = []
for map_name in args.offline_map_lists:
    source_dir = args.offline_data_dir + map_name
    for quality in args.offline_data_quality:
        used_data_dir.append(f"{source_dir}/{quality}/")
buffer.load_offline_data(used_data_dir, args.offline_episode_num, max_epi_length=eval_env.max_timestep)
offline_dataset = buffer.sample()
offline_dataset.stats()
offline_tconf = TrainerConfig(
    max_epochs=1, 
    batch_size=args.offline_mini_batch_size, 
    learning_rate=args.offline_lr,
    num_workers=0, 
    mode="offline",
    use_curiosity=False  
)
offline_trainer = Trainer(model, critic_model, offline_tconf)
target_rtgs = 20.
print("offline target_rtgs: ", target_rtgs)
for i in range(args.offline_epochs):
     offline_actor_loss, offline_critic_loss, _, __, ___ = offline_trainer.train(offline_dataset,
                                                                                 args.offline_train_critic)
     if args.save_log:
         writter.add_scalar('offline/{args.map_name}/offline_actor_loss', offline_actor_loss, i)
         writter.add_scalar('offline/{args.map_name}/offline_critic_loss', offline_critic_loss, i)
     if i % args.offline_eval_interval == 0:
         aver_return, aver_win_rate, _ = rollout_worker.rollout(eval_env, target_rtgs, train=False)
         print("offline epoch: %s, return: %s, eval_win_rate: %s" % (i, aver_return, aver_win_rate))
         if args.save_log:
             writter.add_scalar('offline/{args.map_name}/aver_return', aver_return.item(), i)
             writter.add_scalar('offline/{args.map_name}/aver_win_rate', aver_win_rate, i)
     if args.offline_model_save and i==args.offline_epochs-1:
         actor_path = args.pre_train_model_path + args.exp_name + '/actor'
         if not os.path.exists(actor_path):
             os.makedirs(actor_path)
         critic_path = args.pre_train_model_path + args.exp_name + '/critic'
         if not os.path.exists(critic_path):
             os.makedirs(critic_path)
         torch.save(model.state_dict(), actor_path + os.sep + str(i) + '.pkl')
         torch.save(critic_model.state_dict(), critic_path + os.sep + str(i) + '.pkl')
if args.online_epochs > 0 and args.online_pre_train_model_load:
    actor_path = args.pre_train_model_path + args.exp_name + '/actor/' + str(args.online_pre_train_model_id) + '.pkl'
    critic_path = args.pre_train_model_path + args.exp_name + '/critic/' + str(args.online_pre_train_model_id) + '.pkl'
    model.load_state_dict(torch.load(actor_path), strict=False)
    critic_model.load_state_dict(torch.load(critic_path), strict=False)

online_tconf = TrainerConfig(
    max_epochs=args.online_ppo_epochs, 
    batch_size=0,
    learning_rate=args.online_lr, 
    num_workers=0, 
    mode="online",
    use_lr_scheduler=True,
    use_curiosity=args.use_curiosity,
    feature_dim=args.feature_dim,
    local_obs_dim=local_obs_dim,
    action_dim=action_dim,
    curiosity_lr=args.curiosity_lr,
    intrinsic_reward_coef=args.intrinsic_reward_coef,
    forward_loss_coef=args.forward_loss_coef,
    inverse_loss_coef=args.inverse_loss_coef
)
online_trainer = Trainer(model, critic_model, online_tconf)
buffer.reset(num_keep=0, buffer_size=args.online_buffer_size)
total_steps = 0
rollout_worker.trainer = online_trainer
for i in range(args.online_epochs):
    sample_return, win_rate, steps = rollout_worker.rollout(online_train_env, target_rtgs, train=True)
    total_steps += steps
    online_dataset = buffer.sample()
    online_actor_loss, online_critic_loss, entropy, ratio, confidence = online_trainer.train(online_dataset,
                                                                                             args.online_train_critic)
    if args.save_log:
        writter.add_scalar('online/{args.map_name}/online_actor_loss', online_actor_loss, total_steps)
        writter.add_scalar('online/{args.map_name}/online_critic_loss', online_critic_loss, total_steps)
        writter.add_scalar('online/{args.map_name}/entropy', entropy, total_steps)
        writter.add_scalar('online/{args.map_name}/ratio', ratio, total_steps)
        writter.add_scalar('online/{args.map_name}/confidence', confidence, total_steps)
        writter.add_scalar('online/{args.map_name}/sample_return', sample_return, total_steps)
    print("sample return: %s, online target_rtgs: %s" % (sample_return, target_rtgs))
    # if i % args.online_eval_interval == 0:
    #     print(f"Episode {i}: Win Rate={win_rate:.2f}, Return={sample_return:.2f}, RTG={new_rtg:.2f}")
    #     if writter is not None:
    #         writter.add_scalar("train/rtg", new_rtg, i)
    #         writter.add_scalar("train/win_rate", win_rate, i)
    #         writter.add_scalar("train/return", sample_return, i)
    if i % args.online_eval_interval == 0:
        aver_return, aver_win_rate, _ = rollout_worker.rollout(eval_env, target_rtgs, train=False)
        if online_trainer.config.use_lr_scheduler:
            should_update = False
            if aver_win_rate > online_trainer.config.win_rate_threshold:
                should_update = True
            elif aver_return > target_rtgs * 0.8:  
                should_update = True
            if should_update:
                online_trainer.update_scheduler(aver_win_rate, aver_return)
        print("online steps: %s, return: %s, eval_win_rate: %s" % (total_steps, aver_return, aver_win_rate))
        if args.save_log:
            writter.add_scalar('online/{args.map_name}/aver_return', aver_return.item(), total_steps)
            writter.add_scalar('online/{args.map_name}/aver_win_rate', aver_win_rate, total_steps)
online_train_env.real_env.close()
eval_env.real_env.close()
