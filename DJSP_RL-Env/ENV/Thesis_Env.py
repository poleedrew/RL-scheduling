import gym
from gym.utils import EzPickle
from gym.spaces import Box, Dict, Discrete
from ENV.utils.DynamicJSP import DynamicJobShopInstance
from ENV.DJSP_Env import DJSP_Env
from ENV.utils.PaperRule import Rule1, Rule2, Rule3, Rule4, Rule5, Rule6
from ENV.utils.CommonHeuristic import EDD, SPT, LPT, SRPT, LS, FIFO
import numpy as np 
from gym import spaces
from utils import json_to_dict
import os

def findMaxInList(L):
    res = 0
    for t in L:
        res = res if len(t) == 0 else max(res, max(t))
    return res

class Thesis_Env(DJSP_Env):
    def __init__(self, env_config):
        self.djspArgs = json_to_dict(env_config['djspArgsFile'])
        super().__init__(env_config)
    
        self.__fixed_case = False if 'fixed_case' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['fixed_case']
        self.noop = False if 'noop' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['noop']
        self.maxJobTypeNum = 5 if 'maxJobTypeNum' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['maxJobTypeNum']
        self.maxJobTypeNum = 5 if 'maxJobTypeNum' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['maxJobTypeNum']
        self.maxOPperJob = 5 if 'maxOPperJob' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['maxOPperJob'] 
        self.maxMachineNum = 5 if 'maxMachineNum' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['maxMachineNum'] 
        self.observation_space = spaces.Box(low=np.float32(-128.), high=np.float32(128.), dtype=np.float32, 
                            shape=(7 + self.maxJobTypeNum * 2 * 3,))
        
        print('init done')
        print('max job type num: ', self.maxJobTypeNum)

    def reward(self, state):
        # state means S(t+1) in the paper
        prevU_avg, _, _, _, _, prevTard_e, prevTard_a = self.prevState[0:7]
        U_avg, _, _, _, _, Tard_e, Tard_a = state[0:7]
        if Tard_a < prevTard_a:
            reward = 1
        else:
            if Tard_a > prevTard_a:
                reward = -1
            else:
                if Tard_e < prevTard_e:
                    reward = 1
                else:
                    if Tard_e > prevTard_e:
                        reward = -1
                    else:
                        if U_avg > prevU_avg:
                            reward = 1
                        else:
                            if U_avg >= prevU_avg * 0.95:
                                reward = 0
                            else:
                                reward = -1


        return reward

    def currentState(self):
        U_avg, U_std = self.DJSP_Instance.Machine_UtilizationInfo()
        CRO = self.DJSP_Instance.OP_CompletionRate()
        CRJ_avg, CRJ_std = self.DJSP_Instance.Job_CompletionInfo()
        Tard_e = self.DJSP_Instance.estimatedTardinessRate()
        Tard_a = self.DJSP_Instance.actualTardinessRate()
        state = [
            U_avg,
            U_std,
            CRO,
            CRJ_avg,
            CRJ_std,
            Tard_e,
            Tard_a
        ]
        # job type completion rate (avg, std)
        currentTime = self.DJSP_Instance.currentTime
        job_type_completion = [[] for _ in range(self.maxJobTypeNum)]
        for job in self.DJSP_Instance.Jobs:
            if job.arriveAt(currentTime) and job.isDone() is False:
                job_type_completion[job.job_type].append(job.completionRate_time())
        for info in job_type_completion:
            state += [0., 0.] if len(info) == 0 else [np.mean(info), np.std(info)]
        
        # job type estimated tardy info (avg, std)
        # normalize by max tardy
        estimatedTardy = [[] for _ in range(self.maxJobTypeNum)]
        for job in self.DJSP_Instance.Jobs:
            if job.arriveAt(currentTime) and job.isDone() is False:
                estimatedTardy[job.job_type].append(job.estimatedTardiness(currentTime))
        denominator = findMaxInList(estimatedTardy)
        denominator = 1 if denominator == 0 else denominator # prevent divide by 0
        for info in estimatedTardy:
            state += [0., 0.] if len(info) == 0 else [np.mean(info)/denominator, np.std(info)/denominator]
        
        # job type actual tardy info (avg, std)
        
        actualTardy = [[] for _ in range(self.maxJobTypeNum)]
        for job in self.DJSP_Instance.Jobs:
            if job.arriveAt(currentTime) and job.isDone() is False:
                actualTardy[job.job_type].append(job.actualTardiness(currentTime))

        denominator = findMaxInList(actualTardy)
        denominator = 1 if denominator == 0 else denominator # prevent divide by 0
        for info in actualTardy:
            state += [0., 0.] if len(info) == 0 else [np.mean(info)/denominator, np.std(info)/denominator]
       
        return np.array(state)

if __name__ == '__main__':
    env_config = {
        "djspArgsFile": './args.json',
        "noop": False
    }
    env = Thesis_Env(env_config)
    env.DJSP_Instance.show_registedJobs()
    env.reset()
    done = False
    while not done:
        a = np.random.randint(env.action_space.n)
        _, _, done, info = env.step(a)
    print('done')

    # import ray.rllib.agents.dqn as dqn
    # agent = dqn.DQNTrainer(env=Thesis_Env)
    # checkpoint = './ray_results/Ours_CR_JT_Basic_Rule_Deterministic_Case13_Loose/DQN_thesis_env_a8ae8_00000_0_2021-12-14_01-11-06/checkpoint_000500/checkpoint-500'
    # agent.restore(checkpoint)

    # config = {
    #     'env_config': env_config,
    #     'explore': False
    # }

    # env = Thesis_Env(config, env_config)
    # state = env.reset()
    # while True:
    #     action = agent.compute_action(state)
    #     state, reward, done, info = env.step(action)
    #     if done:
    #         break