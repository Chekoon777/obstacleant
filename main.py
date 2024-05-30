import gymnasium as gym
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, EvalCallback
import os
import argparse

import torch

from customant.customant import CustomAntEnv

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env):

    policy_kwargs = {
        "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        "activation_fn": torch.nn.ReLU,
    }

    if args.sb3_algo == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cuda', policy_kwargs=policy_kwargs, tensorboard_log=log_dir)
    elif args.sb3_algo == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, device='cuda', policy_kwargs=policy_kwargs, tensorboard_log=log_dir)
    elif args.sb3_algo == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, device='cuda', policy_kwargs=policy_kwargs, tensorboard_log=log_dir)
    else:
        print('Algorithm not found')
        return

    """ # Stop training when mean reward reaches reward_threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)

    # Stop training when model shows no improvement after max_no_improvement_evals, 
    # but do not start counting towards max_no_improvement_evals until after min_evals.
    # Number of timesteps before possibly stopping training = min_evals * eval_freq (below)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10000, verbose=1)

    eval_callback = EvalCallback(
        env, 
        eval_freq=10000, # how often to perform evaluation i.e. every 10000 timesteps.
        callback_on_new_best=callback_on_best, 
        callback_after_eval=stop_train_callback, 
        verbose=1, 
        best_model_save_path=os.path.join(model_dir, f"{args.sb3_algo}"),
    ) """

    TIMESTEPS = 25000
    iters = 0
    while iters < 20:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, tb_log_name=f"{args.sb3_algo}", reset_num_timesteps=False)
        model.save(f"{model_dir}/{args.sb3_algo}_{TIMESTEPS*iters}")

def test(env):

    if args.sb3_algo == 'SAC':
        model = SAC.load(os.path.join(model_dir, f"{args.test}.zip"), env=env)
    elif args.sb3_algo == 'TD3':
        model = TD3.load(os.path.join(model_dir, f"{args.test}.zip"), env=env)
    elif args.sb3_algo == 'PPO':
        model = PPO.load(os.path.join(model_dir, f"{args.test}.zip"), env=env)
    else:
        print('Algorithm not found')
        return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3, PPO')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()


    if args.train:
        gymenv = CustomAntEnv(render_mode=None)
        train(gymenv)

    if args.test:
        if os.path.isfile(os.path.join(model_dir, f"{args.test}.zip")):
            gymenv = CustomAntEnv(render_mode='human')
            test(gymenv)
        else:
            print(f'{os.path.join(model_dir, f"{args.test}.zip")} not found.')