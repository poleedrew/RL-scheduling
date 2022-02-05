from tabnanny import check
import numpy as np 
import matplotlib.pyplot as plt
# from ENV.Thesis_Env import Thesis_Env
from ENV.DJSP_Env import DJSP_Env
import ray
from ray.tune.registry import register_env
from ray import tune
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
from ray.rllib.models import ModelCatalog
# from Network.MaskModel import TorchParametricActionsModel
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from typing import Dict
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.examples.env.multi_agent import BasicMultiAgent

from utils import json_to_dict, custom_log_creator, extractProcessTimeFlag
import os
import argparse
import glob

# from ray.rllib.env import PettingZooEnv
# from pettingzoo.classic import leduc_holdem_v2


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
    # metrics["foo"] = 1

    print(metrics)
    return metrics

def train_BJTH(args):
    env_config = {
        # "djspArgsFile": '/home/ubuntu/DJSP_RL/args.json',
        # "djspArgsFile": '/mnt/nfs/work/oo12374/JSP/origin/DJSP_RL-Env/args.json',
        "djspArgsFile": args.args_json,
        "noop": False
    }
    eval_fn = custom_evaluation
    config = {
        "env": "djsp_env",
        "env_config": env_config,# json_to_dict(env_config['djspArgsFile']),
        "framework": "torch",
        "model":{
            "fcnet_hiddens": [30,30,30, 30, 30],
            "fcnet_activation": "tanh",
        },
        "lr": 0.0002,
        "lr_schedule": [
            [0, 0.0002],
            [1000*200, 0.0001]
        ],
        "num_gpus": 4,
        "num_workers": 20, 
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.9,
            "final_epsilon": 0.1,
            "epsilon_timesteps": 1000 * 100,  # Timesteps over which to anneal epsilon.

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        # "timesteps_per_iteration": 200,
        "target_network_update_freq": 20,
        "buffer_size": 10000,
        # "learning_starts": 1000,
        "train_batch_size": 64,
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
    if args.test_only:
        test_ray(env_config, config, args)
    else:
        train_ray(env_config, config, args)
    # for _ in range(10):
    #     env.reset()
    #     env.DJSP_Instance.show_registedJobs()

def env_creator(env_config):
    return DJSP_Env(env_config)  # return an env instance
    # return Foo_Env2(env_config)


def test_ray(env_config, config, args):
    validate_cases = glob.glob('{}/validate_*.json'.format(args.job_type_file))
    validate_cases.sort()
    
    for case in validate_cases:
        best_checkpoint = None
        best_tardiness = 1 << 20
        env = DJSP_Env(env_config=env_config)
        env.load_instance(case)
        print(args.checkpoint)
        for d in os.listdir(args.checkpoint):
            if not ('DQN' in d):
                continue
            print(d)
            for t in range(380, 420+1, 10):
                checkpoint_fname = 'checkpoint_000{}/checkpoint-{}'.format(t, t)
                checkpoint_path = os.path.join(args.checkpoint, d)
                checkpoint_path = os.path.join(checkpoint_path, checkpoint_fname)
                print('checkpoint_path:', checkpoint_path)
                agent = dqn.DQNTrainer(config=config, env=DJSP_Env)
                try:
                    agent.restore(checkpoint_path)
                except FileNotFoundError:
                    break
                episode_reward = 0
                done = False
                obs = env.reset()
                while not done:
                    action = agent.compute_action(obs)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                tardiness = env.DJSP_Instance.Tardiness()
                if tardiness < best_tardiness:
                    best_checkpoint = checkpoint_path
                    best_tardiness = tardiness
            print('testing case:', case, '\tcheckpoint:', best_checkpoint, '\tbest_tardiness:', best_tardiness)
        

def train_ray(env_config, config, args):
    # global CASE_DIR
    # CASE_DIR = json_to_dict(env_config['djspArgsFile'])['JobTypeFile'].split('/')[-2]
    CASE_DIR = args.job_type_file
    
    # arg_dict = json_to_dict(env_config['djspArgsFile'])
    arg_dict = json_to_dict(args.args_json)
    processTimeFlag = extractProcessTimeFlag(arg_dict['RPT_effect'])
    if arg_dict['ENV']['basic_rule'] is True:
        RuleFlag = 'Basic_Rule'
    else:
        RuleFlag = 'BJTH_Rule'

    print(CASE_DIR)
    print(processTimeFlag)
    print(RuleFlag)

    ray.init()
    register_env("djsp_env", env_creator)

    stop = {
        "training_iteration": 500,
    }

    results = tune.run("DQN", 
        # MyTrainableClass,
        config=config, 
        stop=stop,
        checkpoint_freq=10,
        # local_dir="~/ray_results", 
        # local_dir="./ray_results", 
        # local_dir="./results", 
        local_dir="./results_2", 
        # name="{}_{}_{}_{}_{}".format(args.experiment_name, RuleFlag, processTimeFlag, CASE_DIR, args.DDT_type), 
        name="{}".format(args.experiment_name), 
        # keep_checkpoints_num=1, checkpoint_at_end=True
        )
    ray.shutdown()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BJTH program')
    ### Train
    parser.add_argument('--experiment_name', type=str, default='Ours_CR_JT', help='experiment name')
    parser.add_argument('--args_json', type=str, default='args.json', help='argument file')
    parser.add_argument('--job_type_file', type=str, default='./test_instance/Case13', help='validate case directory')
    ### Test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--checkpoint', type=str, help='checkpoint for testing')
    args = parser.parse_args()
    # global CASE_DIR
    # CASE_DIR = args.job_type_file
    train_BJTH(args)