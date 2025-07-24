import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--video", action="store_true", default=False, help="test")
parser.add_argument("--video_interval", type=int, default=250000, help="test")
parser.add_argument("--video_length", type=int, default=400, help="test")
parser.add_argument("--num_envs", type=int, default=400, help="test")
parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0", help="test")
parser.add_argument("--seed", type=int, default=None, help="test")
parser.add_argument("--max_iterations", type=int, default=None, help="test")
parser.add_argument("--continue_train", action="store_true", default=False, help="test")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--log_time", type=str, default=None, help="time of log / policy name.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# returning the argument from commandline and remaining argument go to hydra

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args # adding list to list

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml, load_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from utils import get_checkpoint_path, get_log_time_path
from termcolor import colored, cprint
"""
gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

"""

"""
from "sb3_cfg_entry_point" and args_cli.task, "Isaac-Cartpole-v0"
hydra_task_config find the configulation file "{agents.__name__}:sb3_ppo_cfg.yaml",
gives it to main as agent_cfg

from args_cli.task, "Isaac-Cartpole-v0", hydra_task_config gives 
"env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg", as
env_cfg to the main
"""


def main():
    config_dir = os.path.abspath(os.path.join("../config", "sb3_agent.yaml"))
    env_cfg = CartpoleEnvCfg()
    agent_cfg = load_yaml(config_dir)

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene_num_envs

    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("../logs", "sb3", args_cli.task))
    print(f"[info] Logging in directory: {log_root_path}")
    print(f"Exact experiment name from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    dump_yaml(os.path.join(log_dir, "params", "env_taml"), env_cfg) # save configue as yaml file
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg) # save configue as pickle file
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    agent_cfg = process_sb3_cfg(agent_cfg) # convert yaml type to stable base line classes / components
    policy_arch = agent_cfg.pop("policy") # agent cfg is dict
    n_timesteps = agent_cfg.pop("n_timesteps")

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv): # check if env.unwrapped is an instance of DirectMAREnv class
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0, # ??????????
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[info] recording videos during training")
        print_dict(video_kwargs, nesting=4)
        env=gym.wrappers.RecordVideo(env, **video_kwargs)
        # ** means unpack dict
        # example: video_kwargs = {test=True, str1="test"}
        # env=gym.wrappers.RecordVideo(env, test=Ture, str1="test")

    """
    Sb3VecEnvWrapper() should be perform at the end because this wrapper modify the env 
    to the one that is not compatible with gymnasium.Env
    """
    env = Sb3VecEnvWrapper(env)

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs = "normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            # this line does
                # if "normalize_input" in agent_cfg:
                #     norm_obs = agent_cfg.pop("normalize_input")
                # else:
                #     norm_obs = False  # or None

            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs = "clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    if args_cli.continue_train:
        # directory for logging into
        log_root_path = os.path.join("../logs", "sb3", args_cli.task)
        if args_cli.log_time is not None:
            log_time_path = get_log_time_path(log_root_path, args_cli.log_time)
        else:
            log_time_path = get_log_time_path(log_root_path)
        log_time_path = os.path.abspath(log_time_path)

        # checkpoint and log_dir stuff
        if args_cli.checkpoint is None:
            if args_cli.use_last_checkpoint:
                checkpoint = "model_.*.zip"
            else:
                checkpoint = "model.zip"
            checkpoint_path = get_checkpoint_path(log_time_path, checkpoint)
        else:
            checkpoint_path = os.path.join(log_time_path, args_cli.checkpoint)

        # create agent from stable baselines
        cprint(f"Loading checkpoint from: {checkpoint_path}", 'white', 'on_red')
        agent = PPO.load(checkpoint_path, env, print_system_info=True, reset_num_timesteps=False)
        # to continue learning from check points, it must has reset_num_timesteps=False
    else:
        agent = PPO(policy_arch, env, verbose=1, **agent_cfg)


    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    save_freq =(50) # number of check points will be saved in total
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix="model", verbose=2) # what is this verbose level
    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    agent.save(os.path.join(log_dir, "model"))

    env.close()

if __name__=="__main__":
    main()
    simulation_app.close()