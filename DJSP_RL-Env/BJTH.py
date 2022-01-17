import numpy as np 
import matplotlib.pyplot as plt
from ENV.DJSP_Env import DJSP_Env
import ray
from ray.tune.registry import register_env
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from Network.MaskModel import TorchParametricActionsModel
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from typing import Dict
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.examples.env.multi_agent import BasicMultiAgent
import argparse

from utils import json_to_dict, custom_log_creator, extractProcessTimeFlag

import os
# from ray.rllib.env import PettingZooEnv
# from pettingzoo.classic import leduc_holdem_v2

CASE_DIR = './test_instance/Case4'

class MyTrainableClass(tune.Trainable):
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))

class MyCallbacks(DefaultCallbacks):
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        # assert episode.length > 0, \
        #     "ERROR: `on_episode_step()` callback should not be called right " \
        #     "after env reset!"
        
        # print(f"last info for agent : {episode.last_info_for()}")
        pass

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        # Make sure this episode is really done.
        # assert episode.batch_builder.policy_collectors[
        #     "default_policy"].buffers["dones"][-1], \
        #     "ERROR: `on_episode_end()` should only be called " \
        #     "after episode is done!"
        # pole_angle = np.mean(episode.user_data["pole_angles"])
        # print("episode {} (env-idx={}) ended with length {} and pole "
        #       "angles {}".format(episode.episode_id, env_index, episode.length,
        #                          pole_angle))
        for k, v in episode.last_info_for().items():
            episode.custom_metrics[k] = v
        # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

def custom_evaluation(trainer, eval_workers):
    print('custom_evaluation')
    worker_1, worker_2, worker_3, worker_4, worker_5 = eval_workers.remote_workers()

    # Set different env settings for each worker. Here we use a fixed config,
    # which also could have been computed in each worker by looking at
    # env_config.worker_index (printed in SimpleCorridor class above).
    # worker_1.foreach_env.remote(lambda env: env.set_eval(3))
    # worker_2.foreach_env.remote(lambda env: env.set_eval(7))
    # global CASE_DIR
    CASE_DIR = args.job_type_file
    worker_1.foreach_env.remote(lambda env: env.load_evaluation(os.path.join('./test_instance', CASE_DIR, 'validate_1.json')))
    worker_2.foreach_env.remote(lambda env: env.load_evaluation(os.path.join('./test_instance', CASE_DIR, 'validate_2.json')))
    worker_3.foreach_env.remote(lambda env: env.load_evaluation(os.path.join('./test_instance', CASE_DIR, 'validate_3.json')))
    worker_4.foreach_env.remote(lambda env: env.load_evaluation(os.path.join('./test_instance', CASE_DIR, 'validate_4.json')))
    worker_5.foreach_env.remote(lambda env: env.load_evaluation(os.path.join('./test_instance', CASE_DIR, 'validate_5.json')))
    

    for i in range(5):
        print("Custom evaluation round", i)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    # You can also put custom values in the metrics dict.
    metrics["foo"] = 1
    print(metrics)
    return metrics

def train_BJTH():
    env_config = {
        # "djspArgsFile": '/home/ubuntu/DJSP_RL/args.json',
        "djspArgsFile": '/mnt/nfs/work/oo12374/JSP/origin/DJSP_RL-Env/args.json',
        "noop": False
    }
    test_ray(env_config)
    # for _ in range(10):
    #     env.reset()
    #     env.DJSP_Instance.show_registedJobs()

def env_creator(env_config):
    return DJSP_Env(env_config)  # return an env instance
    # return Foo_Env2(env_config)

def test_ray(env_config):
    # global CASE_DIR
    # CASE_DIR = json_to_dict(env_config['djspArgsFile'])['JobTypeFile'].split('/')[-2]
    CASE_DIR = args.job_type_file
    
    # arg_dict = json_to_dict(env_config['djspArgsFile'])
    arg_dict = json_to_dict(args.args_json)
    # processTimeFlag = 'Deterministic' if arg_dict['RPT_effect'] is False else 'Stochastic'
    processTimeFlag = extractProcessTimeFlag(arg_dict['RPT_effect'])
    if arg_dict['ENV']['basic_rule'] is True:
        RuleFlag = 'Basic_Rule'
    else:
        RuleFlag = 'BJTH_Rule'

    print(CASE_DIR)
    print(processTimeFlag)
    print(RuleFlag)

    ray.init()
    register_env("bjth_env", env_creator)

    eval_fn = custom_evaluation
    config = {
        "env": "bjth_env",
        "env_config": json_to_dict(env_config['djspArgsFile']),
        "framework": "torch",
        "model":{
            "fcnet_hiddens": [30, 30, 30, 30, 30],
            "fcnet_activation": "tanh",
        },
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.5,
            "final_epsilon": 0.1,
            "epsilon_timesteps": 1000 * 100,  # Timesteps over which to anneal epsilon.

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        "target_network_update_freq": 20,
        "buffer_size": 10000,
        # "learning_starts": 1000,
        "train_batch_size": 64,
        "lr": 0.0002,
        "lr_schedule": [
            [0, 0.0002],
            [1000*200, 0.0001]
        ],
        "num_gpus": 4,
        "num_workers": 20, 
        # "num_envs_per_worker": 2,
        "callbacks": MyCallbacks,
        "evaluation_num_workers": 5,
        # Optional custom eval function.
        "custom_eval_function": eval_fn,
        # Enable evaluation, once per training iteration.
        "evaluation_interval": 10,
        # Run 10 episodes each time evaluation runs.
        "evaluation_num_episodes": 10,
        # Override the env config for evaluation.
        "evaluation_config": {
            'env_config': env_config,
            'explore': False
        }
    }
    stop = {
        "training_iteration": 500,
    }
    
    results = tune.run("DQN", 
        # MyTrainableClass,
        config=config, 
        stop=stop,
        checkpoint_freq=10,
        local_dir="./ray_results", 
        # name="BJTH_{}_{}_{}".format(RuleFlag,processTimeFlag,CASE_DIR),
        name="{}_{}_{}_{}_{}".format(args.experiment_name, RuleFlag, processTimeFlag, CASE_DIR, args.DDT_type), 
        # keep_checkpoints_num=1, checkpoint_at_end=True
        )
    ray.shutdown()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Thesis program')
    parser.add_argument('--experiment_name', type=str, default='Ours_CR_JT', help='experiment name')
    parser.add_argument('--DDT_type', type=str, default='Loose', help='Due Day Tightness type: Loose, Tight')
    parser.add_argument('--args_json', type=str, default='args.json', help='argument file')
    parser.add_argument('--job_type_file', type=str, default='Case13', help='validate case directory')
    args = parser.parse_args()
    CASE_DIR = args.job_type_file
    # global CASE_DIR
    
    train_BJTH()