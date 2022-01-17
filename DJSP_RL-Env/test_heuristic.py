import numpy as np 
import matplotlib.pyplot as plt
from ENV.DJSP_Env import DJSP_Env
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
# from Args import args
import os, glob
from random import choice
import numpy as np
from ENV.utils.PaperRule import Rule1, Rule2, Rule3, Rule4, Rule5, Rule6
from ENV.utils.CommonHeuristic import EDD, SPT, LPT, SRPT, LS, FIFO, CR
# from Logger import Logger
from utils import json_to_dict
from collections import defaultdict
import argparse

def test_rule(Rules):
    for key, value in Rules.items():
        # LOGGER = Logger(prefix='heuristic_%s'%(key))
        for case in validate_cases:
            env.load_instance(case)
            res = []
            for _ in range(50):
                env.restart()
                env.DJSP_Instance.scheduling(value)
                info = env.DJSP_Instance.check_schedule()
                if info['error code'] != 0:
                    print('[Error] ', info['status'])
                tardiness = env.DJSP_Instance.Tardiness()
                res.append(tardiness)
                # env.DJSP_Instance.show_schedule()
            print("[%7s], [%s], Tardiness: (%10.2f, %10.2f)" % (key, case, np.mean(res), np.std(res)))
            
            # LOGGER.record(case, np.mean(res), 0)
            # LOGGER.record(case, np.mean(res), 100000)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Thesis program')
    parser.add_argument('--args_json', type=str, default='args.json', help='argument file')
    parser.add_argument('--job_type_file', type=str, default='./test_instance/Case13', help='validate case directory')
    args = parser.parse_args()

    # DJSP_config = json_to_dict('./args.json')
    DJSP_config = json_to_dict(args.args_json)
    env = DJSP_Env(
        DJSP_config = DJSP_config,
    )
    # dir_name = './test_instance/Case13'
    dir_name = args.job_type_file
    # env.load_basic_type()
    validate_cases = glob.glob('{}/validate_*.json'.format(dir_name))
    validate_cases.sort()
    print(validate_cases)
    res = defaultdict(list)
    # for case in validate_cases:
    #     env.load_evaluation(case)
    #     for _ in range(100):
    #         obs = env.reset()
    #         done = False
    #         info = {}
    #         while not done:
    #             action = np.random.randint(env.action_space.n)
    #             _, _, done, info = env.step(action)
    #         for k, v in info.items():
    #             res[k].append(v)
    # for k, v in res.items():
    #     print('Random composite, {}, mean: {}, std: {}'.format(k, np.mean(v), np.std(v)))
    rule_config = {
        'machineSelection': 'SPT'
    }
    Rules = {
        'Rule 1': Rule1(),
        'Rule 2': Rule2(),
        'Rule 3': Rule3(),
        'Rule 4': Rule4(),
        'Rule 5': Rule5(),
        'Rule 6': Rule6(),
    }
    CommonRules = {
        'SPT': SPT(rule_config),
        'LPT': LPT(rule_config),
        'SRPT': SRPT(rule_config),
        'LS': LS(rule_config),
        'FIFO': FIFO(rule_config),
        'EDD': EDD(rule_config),
        'CR': CR(rule_config)
    }
    test_rule(CommonRules)
    # test_rule(Rules)
    