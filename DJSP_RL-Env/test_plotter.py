import argparse
import os
import glob

from djsp_plotter import Plotter
from djsp_logger import DJSP_Logger
from ENV.utils.CommonHeuristic import EDD, SPT, LPT, SRPT, LS, FIFO, CR
from ENV.utils.PaperRule import Rule1, Rule2, Rule3, Rule4, Rule5, Rule6
from utils import json_to_dict
from ENV.DJSP_Env import DJSP_Env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Thesis program')
    parser.add_argument('--args_json', type=str, default='args.json', help='argument file')
    parser.add_argument('--validate_dir', type=str, default='./test_instance/Case13', help='validate case directory')
    args = parser.parse_args()
    rule_config = {
        'machineSelection': 'SPT'
    }
    rules = {
        'SPT':      SPT(rule_config),
        'LPT':      LPT(rule_config),
        'SRPT':     SRPT(rule_config),
        'LS':       LS(rule_config),
        'FIFO':     FIFO(rule_config),
        'EDD':      EDD(rule_config),
        'CR':       CR(rule_config),
        'Rule 1':   Rule1(),
        'Rule 2':   Rule2(),
        'Rule 3':   Rule3(),
        'Rule 4':   Rule4(),
        'Rule 5':   Rule5(),
        'Rule 6':   Rule6(),
    }
    validate_dir = args.validate_dir
    validate_cases = glob.glob('{}/validate_*.json'.format(validate_dir))
    out_dir = 'interactive_timeline'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for case in validate_cases:
        tmp, ext = os.path.splitext(case)
        l = tmp.split('/')
        out_case_dir = os.path.join(out_dir, l[-2], l[-1])
        print(out_case_dir)
        if not os.path.exists(out_case_dir):
            os.makedirs(out_case_dir) 
        for rule_name, rule in rules.items():
            DJSP_config = json_to_dict(args.args_json)
            env = DJSP_Env(
                DJSP_config = DJSP_config,
            )
            env.load_instance(case)
            env.restart()
            env.DJSP_Instance.scheduling(rule)
            tardiness = round(env.DJSP_Instance.Tardiness(), 2)
            makespan = round(env.DJSP_Instance.makespan(), 2)
            print('{}_{}\ttardiness: {}, makespan: {}'.format(case, rule_name, tardiness, makespan))
            info = env.DJSP_Instance.check_schedule()
            assert(info['error code'] == 0)
            
            schedule_result_path = '{}.pth'.format(os.path.join(out_case_dir, rule_name))
            html_path = '{}.html'.format(os.path.join(out_case_dir, rule_name))
            env.DJSP_Instance.logger.save(schedule_result_path)
            logger = DJSP_Logger()
            logger.load(schedule_result_path)
            plotter = Plotter(logger)
            plotter.plot_interactive_gantt(html_path)


