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
# from Logger import Logger
from utils import json_to_dict, custom_log_creator, extractProcessTimeFlag
from collections import defaultdict
import argparse

from ENV.utils.CommonHeuristic import EDD, SPT, LPT, SRPT, LS, FIFO, CR
from ENV.utils.PaperRule import Rule1, Rule2, Rule3, Rule4, Rule5, Rule6
from ENV.Thesis_Env import Thesis_Env
from djsp_logger import DJSP_Logger
import ray.rllib.agents.dqn as dqn
from ray import tune


def schedule(env, filename):
    print('schedule')
    env.load_instance(filename)
    env.restart()
    rule_config = {
        'machineSelection': 'SPT'
    }
    env.DJSP_Instance.scheduling(CR(rule_config))
    info = env.DJSP_Instance.check_schedule()
    if info['error code'] != 0:
        print('[Error] ', info['status'])
    tardiness = env.DJSP_Instance.Tardiness()
    makespan = env.DJSP_Instance.makespan()
    env.DJSP_Instance.logger.save('test_scheduling.pth')
    return tardiness, makespan


def reschedule(env, validate_file, logger):
    print('reschedule')    
    env.load_instance(validate_file)
    env.restart()
    for op_info in logger.history:
        if env.DJSP_Instance.All_finished():
            break
        job_id = op_info['job_id']
        op_id = op_info['op_id']
        machine_id = op_info['machine_id']
        env.DJSP_Instance.availableJobs()
        env.DJSP_Instance.assign(job_id, machine_id)
        
    info = env.DJSP_Instance.check_schedule()
    if info['error code'] != 0:
        print('[Error] ', info['status'])
    tardiness = env.DJSP_Instance.Tardiness()
    makespan = env.DJSP_Instance.makespan()
    return tardiness, makespan

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Thesis program')
    parser.add_argument('--args_json', type=str, default='args.json', help='argument file')
    parser.add_argument('--validate_file', type=str, default='./test_instance/Case13/validate_1.json', help='validate case directory')
    args = parser.parse_args()

    arg_dict = json_to_dict(args.args_json)
    processTimeFlag = extractProcessTimeFlag(arg_dict['RPT_effect'])
    print(processTimeFlag)
    DJSP_config = json_to_dict(args.args_json)
    env = DJSP_Env(
        DJSP_config = DJSP_config,
    )
    validate_file = args.validate_file
    print(validate_file)

    tardiness1, makespan1 = schedule(env, validate_file)

    logger = DJSP_Logger()
    logger.load('test_scheduling.pth')
    tardiness2, makespan2 = reschedule(env, validate_file, logger)
    print('tardiness:', tardiness1)
    print('makespan:', makespan1)
    assert(tardiness1 == tardiness2)
    assert(makespan1 == makespan2)

