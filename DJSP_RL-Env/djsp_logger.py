import json
import pickle
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pprint import pprint

class DJSP_Logger(object):
    def __init__(self):
        self.history = []
        self.jobs_to_schedule = []
        self.num_machine = 5
        self.num_job_type = 5
    
    def add_job(self, job):
        job_info = {
            'job_id':           job.job_id,
            'arrival_time':     job.arrival_time, 
            'due_time':         job.due_date, 
            'job_type':         job.job_type,
            'DDT':              job.DDT,
        }
        self.jobs_to_schedule.append(job_info)
        

    def add_op(self, op, machine_id, job_type):
        # add op information to history
        op_info = {
            'job_id':       op.job_id,
            'op_id':        op.op_id,
            'machine_id':   machine_id,
            'startTime':    op.startTime, 
            'RPT':          op.RPT,
            'finishTime':   op.startTime+op.RPT,
            'job_type':     job_type,
        }
        self.history.append(op_info)
    
    def save(self, filename):
        ### save only history
        # with open(filename, 'wb') as f:
        #     pickle.dump(self.history, f)

        ### save as json file
        # json_obj = {
        #     'num_machine':      self.num_machine, 
        #     'num_job_type':     self.num_job_type,
        #     'history':          self.history, 
        #     'jobs_to_schedule': self.jobs_to_schedule
        # }
        # with open(filename, 'wb') as f:
        #     f.write(json_obj)

        ### save by torch.save
        torch.save({
            'num_machine':      self.num_machine, 
            'num_job_type':     self.num_job_type,
            'history':          self.history, 
            'jobs_to_schedule': self.jobs_to_schedule
        }, filename)
        

    def load(self, filename):
        ### load only history
        # with open(filename, 'rb') as f:
        #     self.history = pickle.load(f)

        ### load as json object
        # with open(filename, 'rb') as f:
        #     json_obj = json.load(f)
        # self.num_machine = json_obj['num_machine']
        # self.num_job_type = json_obj['num_job_type']
        # self.history = json_obj['history']
        # self.jobs_to_schedule = json_obj['jobs_to_schedule']

        record = torch.load(filename)
        self.num_machine = record['num_machine']
        self.num_job_type = record['num_job_type']
        self.history = record['history']
        self.jobs_to_schedule = record['jobs_to_schedule']
    
    def to(self, fig_type):        
        if fig_type == 'interactive_gantt_input':
            data = []
            for t, op_info in enumerate(self.history):
                d = dict(
                    Order =         t,
                    Task =          'Machine '+str(op_info['machine_id']), 
                    machine_id =    op_info['machine_id'],
                    Start =         op_info['startTime'],
                    Finish =        op_info['finishTime'],
                    RPT =           op_info['RPT'],
                    job_type =      str(op_info['job_type']),
                    job_id =        op_info['job_id'],
                    op_id =         op_info['op_id'],
                    foo =           1000,
                )
                data.append(d)
            # data = sorted(data, key=lambda op_info: op_info['machine_id'], reverse=True)
            data = sorted(data, key=lambda op_info: op_info['job_type'])
            return pd.DataFrame(data)
        if fig_type == 'scheduling_process_input':
            machine_histories = [[] for _ in range(self.num_machine)]
            for op_info in self.history:
                machine_histories[op_info["machine_id"]].append(op_info)
            return machine_histories

    
    def __str__(self):
        s = ''
        for job_info in self.jobs_to_schedule:
            s += 'job_id: {}, arrival_time: {}, due_time: {}, job_type: {}, DDT: {}\n'.format(
                job_info['job_id'], job_info['arrival_time'], job_info['due_time'], job_info['job_type'], job_info['DDT']
            )
        for op_info in self.history:
            job_id = op_info['job_id']
            machine_id = op_info['machine_id']
            s += 'job_id: {}, machine_id: {}, op_id: {}, op.startTime: {}, op.RPT: {}\n'.format(
                job_id, machine_id, op_info['op_id'], op_info['startTime'], op_info['RPT'])
        return s
    
if __name__ == '__main__':
    logger = DJSP_Logger()
    logger.load('test_scheduling.pth')
    # print(logger)


