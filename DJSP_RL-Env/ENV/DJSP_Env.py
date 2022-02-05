import gym
from gym.utils import EzPickle
from gym.spaces import Box, Dict, Discrete
from ENV.utils.DynamicJSP import DynamicJobShopInstance
from ENV.utils.PaperRule import Rule1, Rule2, Rule3, Rule4, Rule5, Rule6
from ENV.utils.CommonHeuristic import EDD, SPT, LPT, SRPT, LS, FIFO, CR
import numpy as np 
from gym import spaces
from utils import json_to_dict
import os

def write_record(info, action_info):
    f = open('./gant/record.txt', 'w')
    for i in range(6):
        f.write('{}: {}\n'.format(action_info[i], info[action_info[i]]))
    f.close()

class DJSP_Env(gym.Env, EzPickle):
    def __init__(self, env_config):
        self.config = json_to_dict(env_config['djspArgsFile'])
        self.init_instance()
        self.load_basic_type(self.config['JobTypeFile'])
        # self.DJSP_Instance.show_registedJobs()
        self.load_action()
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=np.float32(-128.), high=np.float32(128.), dtype=np.float32, shape=(7,))
        self.evaluation_flag = False
        self.evaluation_file = ''
        self.ruleUsage = [0 for _ in range(self.action_space.n)]
    
    def load_action(self):
        if self.config['ENV']['basic_rule'] is False:
            print('load BJTH rules')
            self.actions = [
                Rule1(), 
                Rule2(), 
                Rule3(), 
                Rule4(), 
                Rule5(), 
                Rule6()
            ]
            self.action_info = {
                0: 'Rule1',
                1: 'Rule2',
                2: 'Rule3',
                3: 'Rule4',
                4: 'Rule5',
                5: 'Rule6',
            }
        else:
            print('load basic rules')
            self.actions = [
                EDD ({'machineSelection': 'SPT'}), 
                SPT ({'machineSelection': 'SPT'}), 
                LPT ({'machineSelection': 'SPT'}), 
                SRPT({'machineSelection': 'SPT'}), 
                LS  ({'machineSelection': 'SPT'}), 
                FIFO({'machineSelection': 'SPT'}),
                CR  ({'machineSelection': 'SPT'}),
            ]
            self.action_info = {
                0: 'EDD',
                1: 'SPT',
                2: 'LPT',
                3: 'SRPT',
                4: 'LS',
                5: 'FIFO',
                6: 'CR',
            }


    def step(self, action):
        # action: np array
        self.ruleUsage[action] += 1
        self.DJSP_Instance.applyRule(self.actions[action])
        currentState = self.currentState()
        reward = self.reward(currentState)
        self.prevState = currentState
        done = self.isDone()
        info = {}
        if done is True and self.evaluation_flag is True:
            info[self.evaluation_file + '_tardiness'] = self.DJSP_Instance.Tardiness()
            info[self.evaluation_file + '_makespan'] = self.DJSP_Instance.makespan()
            info['{}_check'.format(self.evaluation_file)] = self.DJSP_Instance.check_schedule()['error code']
            for idx, usage in enumerate(self.ruleUsage):
                info[self.action_info[idx]] = usage/sum(self.ruleUsage)
            # self.DJSP_Instance.show_schedule('./gant/{}_opt'.format(self.evaluation_file.split('/')[-2]))
            '''
            if info[self.evaluation_file] == 0:
                self.DJSP_Instance.show_schedule('./gant/{}_opt'.format(self.evaluation_file.split('/')[-2]))
                write_record(info, self.action_info)
            '''
            
        return currentState, reward, done, info
        
    def reward(self, state):
        # state means S(t+1) in the paper
        prevU_avg, _, _, _, _, prevTard_e, prevTard_a = self.prevState
        U_avg, _, _, _, _, Tard_e, Tard_a = state
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
    
    def init_instance(self):
        self.DJSP_Instance = DynamicJobShopInstance(
            Machine_num = self.config["MachineNum"],
            Machine_num_dist_op = [1, self.config["MachineNum"] + 1],
            Initial_job_num=self.config["InitialJobNum"],
            Inserted_job_num=self.config["InsertedJobNum"],
            DDT_range=[self.config['DDT'], self.config['DDT'] + self.config['DDT_dt']],
            Job_operation_num_dist=[1, self.config["MaxOpNum"] + 1],
            Process_time_dist=[10, self.config["MaxProcessTime"]],
            RPT_effect=self.config["RPT_effect"],
            Mean_of_arrival=self.config["Mean_of_Arrival"],
        )

    def reset(self):
        # 
        if self.evaluation_flag is False:
            self.DJSP_Instance.restart(refresh = True)
        else:
            self.DJSP_Instance.restart(refresh = False)
        self.prevState = self.currentState()
        self.ruleUsage = [0 for _ in range(self.action_space.n)]
        return self.prevState

    def isDone(self):
        return self.DJSP_Instance.All_finished()

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
        return np.array(state)

    def restart(self, refresh=False):
        self.DJSP_Instance.restart(refresh)
        # self.prevState = self.currentState()
        # return self.prevState

    def load_instance(self, file_name):
        self.DJSP_Instance.load_instance(file_name)
        self.restart()

    def load_basic_type(self, file_name):
        self.DJSP_Instance.load_instance(file_name)
        self.restart(refresh=False)
    
    def load_evaluation(self, file_path):
        self.DJSP_Instance.load_instance(file_path)
        self.evaluation_flag = True
        self.evaluation_file = file_path
        # self.restart(refresh=False)

    def close(self):
        pass

if __name__ == '__main__':
    config = json_to_dict('./args.json')
    env = DJSP_Env(config)
    # env.load_evaluation('./test_instance/Case6/validate_4.json')
    # env.DJSP_Instance.scheduling(EDD({'machineSelection': 'SPT'}))
    # print(env.DJSP_Instance.check_schedule())
    # print('tardiness: ', env.DJSP_Instance.Tardiness())