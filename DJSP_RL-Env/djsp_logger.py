from cmd import IDENTCHARS
import json
import pickle
import sys
import os
from textwrap import indent
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime, timedelta 
from pprint import pprint

class DJSP_Logger(object):
    def __init__(self, num_job=50, num_machine=5, num_job_type=5):
        self.history = []
        self.jobs_to_schedule = []
        self.num_machine = num_machine
        self.num_job_type = num_job_type
        self.num_job = num_job
        self.timestamp = 0
        self.db_path = os.path.join('loggerDB', datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f") +'.db') 
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            '''CREATE TABLE LOGGER
            (TIMESTAMP  INT     PRIMARY KEY     NOT NULL,
            MACHINE_ID  INT     NOT NULL,
            JOB_TYPE    INT     NOT NULL,
            JOB_ID      INT     NOT NULL,
            OP_ID       INT     NOT NULL,
            START_TIME  REAL    NOT NULL,
            FINISH_TIME REAL    NOT NULL, 
            RPT         REAL    NOT NULL
            );''')
    
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
            'timestamp':    self.timestamp,
            'job_id':       op.job_id,
            'op_id':        op.op_id,
            'machine_id':   machine_id,
            'start_time':    op.startTime, 
            'RPT':          op.RPT,
            'finish_time':   op.startTime+op.RPT,
            'job_type':     job_type,
        }
        self.timestamp += 1
        self.history.append(op_info)
        print(op_info)
        self.cursor.execute(
                "INSERT INTO LOGGER (TIMESTAMP, MACHINE_ID, JOB_TYPE, JOB_ID, OP_ID, START_TIME, FINISH_TIME, RPT) \
                VALUES (%d, %d, %d, %d, %d, %f, %f, %f)" %(
                    op_info['timestamp'], op_info['machine_id'], op_info['job_type'], op_info['job_id'], op_info['op_id'], 
                    op_info['start_time'], op_info['finish_time'], op_info['RPT']))
    
    def save(self, filename):
        ### save by torch.save
        torch.save({
            'num_machine':      self.num_machine, 
            'num_job_type':     self.num_job_type,
            'history':          self.history, 
            'jobs_to_schedule': self.jobs_to_schedule
        }, filename)
        self.conn.commit()
        self.conn.close()

    def load(self, filename, db_path=None):
        record = torch.load(filename)
        self.num_machine = record['num_machine']
        self.num_job_type = record['num_job_type']
        self.history = record['history']
        self.jobs_to_schedule = record['jobs_to_schedule']
        if db_path != None:
            self.conn = sqlite3.connect(db_path)
    
    def to(self, fig_type):        
        if fig_type == 'interactive_gantt_input':
            unix_epoch = datetime.strptime('1970-01-01', '%Y-%m-%d')
            data = []
            for t, op_info in enumerate(self.history):
                d = dict(
                    Order =         t,
                    Task =          'Machine '+str(op_info['machine_id']), 
                    machine_id =    op_info['machine_id'],
                    Start =         op_info['start_time'],
                    Finish =        op_info['finish_time'],
                    StartDateTime = unix_epoch + timedelta(days=op_info['start_time']),
                    FinishDateTime = unix_epoch + timedelta(days=op_info['finish_time']),
                    RPT =           op_info['RPT'],
                    job_type =      str(op_info['job_type']),
                    job_id =        op_info['job_id'],
                    op_id =         op_info['op_id'],
                    foo =           t % 5,
                )
                data.append(d)
            # data = sorted(data, key=lambda op_info: op_info['job_type'], reverse=True)
            # print('data:', data)
            # data = sorted(data, key=lambda op_info: op_info['Order'])
            # data = sorted(data, key=lambda op_info: op_info['job_id'])
            return data
        if fig_type == 'scheduling_process_input':
            machine_histories = [[] for _ in range(self.num_machine)]
            for op_info in self.history:
                machine_histories[op_info["machine_id"]].append(op_info)
            return machine_histories
        if fig_type == 'naive_gantt_input':
            # (TIMESTAMP, MACHINE_ID, JOB_TYPE, JOB_ID, OP_ID, START_TIME, FINISH_TIME, RPT)
            machine_histories = []
            cmap = plt.cm.get_cmap('hsv', self.num_job + 1)
            for machine_id in range(self.num_machine):
                segment = []
                color = []
                cursor = self.conn.execute(
                    "SELECT TIMESTAMP, MACHINE_ID, JOB_TYPE, JOB_ID, OP_ID, START_TIME, FINISH_TIME, RPT from LOGGER \
                    WHERE MACHINE_ID==%d " %(machine_id))
                
                for row in cursor:
                    print("TIMESTAMP:{}, MACHINE_ID:{}, JOB_TYPE:{}, JOB_ID:{}, OP_ID:{}, START_TIME:{}, FINISH_TIME:{}, RPT:{}".format(
                        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], )
                    )
                    job_type = row[2]
                    job_id = row[3]
                    start_time = row[5]
                    RPT = row[7]
                    segment.append((start_time, RPT))
                    color.append(cmap(job_id))
                machine_histories.append((segment, color))
            return machine_histories
    def radiantQ_json(self):
        unix_epoch = datetime.strptime('2020-01-01T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
        data = []
        for t, op_info in enumerate(self.history):
            # if t >= 3:
            #     break
            start_time = unix_epoch + timedelta(hours=op_info['start_time'])
            d = {
                # "Name":         "Task " + str(t),
                "Name":         "Job%d, Op%d, Machine%d, Start time:%f, RPT:%f" %(
                    op_info['job_id'], op_info['op_id'], op_info['machine_id'], op_info['start_time'], op_info['RPT']),
                "ID":           t,
                "SortOrder":    t,
                "StartTime":    str(start_time.strftime('%Y-%m-%dT%H:%M:%SZ')),
                "Effort":       str(int(op_info['RPT'])) + ":00:00",
            }
            data.append(d)
        with open('sample.json', 'w') as f:
            json.dump(data, f, indent=4)
    
    def google_chart_gantt_input(self, input_file_name):
        data = []
        for t, op_info in enumerate(self.history):
            interval = [
                'Machine %d' %(op_info['machine_id']), 
                'Job%d, Op%d, Start time:%f, RPT:%f, Finish Time:%f' %(
                    op_info['job_id'], op_info['op_id'], 
                    op_info['start_time'], op_info['RPT'], op_info['finish_time']),
                'new Date(%d,0,0)' %(op_info['start_time']),
                'new Date(%d,0,0)' %(op_info['finish_time'])
            ]
            machine_str = 'Machine %d' %(op_info['machine_id'])
            info_str = 'Job%d, Op%d, Start time:%f, RPT:%f, Finish Time:%f' %(
                    op_info['job_id'], op_info['op_id'], 
                    op_info['start_time'], op_info['RPT'], op_info['finish_time']),
            start_str = 'new Date(%d,0,0)' %(op_info['start_time'])
            finish_str = 'new Date(%d,0,0)' %(op_info['finish_time'])
            
            data.append(interval)
        with open(input_file_name, "w") as f:
            json.dump(str(data), f)

    def __str__(self):
        s = ''
        for job_info in self.jobs_to_schedule:
            s += 'job_id: {}, arrival_time: {}, due_time: {}, job_type: {}, DDT: {}\n'.format(
                job_info['job_id'], job_info['arrival_time'], job_info['due_time'], job_info['job_type'], job_info['DDT']
            )
        for op_info in self.history:
            job_id = op_info['job_id']
            machine_id = op_info['machine_id']
            s += 'machine_id: {}, job_type: {}, job_id: {}, op_id: {}, op.start_time: {}, op.finish_time: {}, op.RPT: {}\n'.format(
                machine_id, op_info['job_type'], job_id, op_info['op_id'], op_info['start_time'], op_info['finish_time'], op_info['RPT'])
        return s
    
if __name__ == '__main__':
    logger = DJSP_Logger()
    logger.load('test_scheduling.pth')
    # print(logger)


