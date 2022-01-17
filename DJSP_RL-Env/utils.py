import numpy as np
import os
import tempfile
from datetime import datetime

def print_dict(d):
    for key, value in d.items():
        print("{}: {}".format(key, value))

def json_to_dict(file_name):
    import json
    print(os.path.join(os.getcwd(), file_name))
    with open(os.path.join(os.getcwd(), file_name), 'r') as fp:
        data = json.load(fp)
    return data

def extractProcessTimeFlag(RPT_effect):
    if RPT_effect['flag'] is False:
        return 'Deterministic'
    else:
        return RPT_effect['type']

def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator
class Machine:
    def __init__(self, machine_id, RPT_effect):
        self.machine_id = machine_id
        self.currentTime = 0
        self.RPT_effect = RPT_effect
        self.history = []
    
    def processOP(self, op):
        machineTime = self.machineTime()
        # print('op["currentTime"]: {}, machineTime: {}'.format(op["currentTime"], machineTime))
        startTime = max(op["currentTime"], machineTime)
        op["startTime"] = startTime
        op["RPT"] = self.real_process_time(op["process_time"][self.machine_id])
        finishedTime = op["startTime"] + op["RPT"] 
        self.history.append(op)
        # print("[Machine] Job {}, op {}, s: {}, e: {}".format(op["job_type"], op["op_id"], op["startTime"], finishedTime))
        return finishedTime

    def real_process_time(self, EPT):
        if self.RPT_effect['flag'] is False:
            # print('EPT')
            return EPT
        else:
            if self.RPT_effect['type'] == 'Gaussian':
                std = EPT / 10.0
                RPT = np.random.normal(EPT, std)
                return np.clip(RPT, EPT - 3 * std, EPT + 3 * std)
            elif self.RPT_effect['type'] == 'Rework':
                if np.random.rand() <= self.RPT_effect['rework_probability']:
                    # rework
                    # print('rework')
                    RPT = EPT + EPT * self.RPT_effect['rework_percentage']
                else:
                    RPT = EPT
                return RPT
            else:
                print('no register RPT effect')
            

    def machineTime(self):
        if len(self.history) == 0:
            return 0
        else:
            return self.history[-1]["startTime"] + self.history[-1]["RPT"]

    def Workload(self):
        return sum(op['RPT'] for op in self.history)

    def utilizationRate(self):
        if self.machineTime() == 0:
            return 0
        else:
            return self.Workload() / self.machineTime()

    def process_history(self, cmap):
        data, color, op_id = [], [], []
        # print("Machine {}".format(self.machine_id))
        for op in self.history:
            data.append((op["startTime"], op["RPT"]))
            # color.append(cmap(op["job_type"]))
            color.append(cmap(op["job_id"]))
            op_id.append(op["op_id"])
            # print("{}, {}, E:{}, R:{}".format(op.job_id, op.op_id,op.EPT,  op.RPT))
            
        return data, tuple(color), op_id

    def validateSchedule(self):
        t = 0
        for op in self.history:
            if t > op.startTime:
                return False
            t = op.completionTime()
        return True

    def reset(self):
        self.current_time, self.history = 0, []

    def show_history(self):
        print("History: ")
        for op in self.history:
            op.show()     

    def check_invalid(self):
        for op in self.history:
            if op["process_time"][self.machine_id] <= 0:
                print("[Invalid] Assign OP({}, {}) on machine {}".format(op["job_type"], op["op_id"], self.machine_id))
                return False
        return True

class Job:
    def __init__(self, 
        job_id,
        job_type,
        arrival_time,
        DDT,
        op_config
    ):
        self.job_id = job_id
        self.job_type = job_type
        self.arrival_time = arrival_time
        self.completion_time = -1
        self.DDT = DDT              # Due Date Tardiness
        self.config = op_config
        self.operations = [Operation(self.job_id, self.arrival_time, config) for config in op_config]
        self.currentOpID = 0
        self.TPT = sum(op.meanEPT for op in self.operations) # Total Process Time
        self.FPT = 0                                     # Finished Process Time
        self.due_date = self.arrival_time + self.DDT * self.TPT
        # print("???",self.due_date)
        
    def updateCurrentOP(self, initTime):
        self.operations[self.currentOpID].initTime = initTime

    def currentOP(self):
        if self.currentOpID == -1:
            return None
        else:
            return self.operations[self.currentOpID] 

    def completionRate(self):
        if self.currentOpID == -1:
            return 1
        else:
            return self.currentOpID / len(self.operations)

    def completedProcessNum(self):
        return len(self.operations) if self.currentOpID == -1 else self.currentOpID

    def remainProcessNum(self):
        return 0 if self.currentOpID == -1 else len(self.operations) - self.currentOpID

    def remainProcessTime(self):
        if self.currentOpID == -1:
            return 0
        else:
            return sum(op.meanEPT for op in self.operations[self.currentOpID:])

    def remain_process_info(self):
        if self.currentOpID == -1:
            return 0, 0
        else:
            return sum(op.meanEPT for op in self.operations[self.currentOpID:]), len(self.operations) - self.currentOpID

    def estimatedTardiness(self, currentTime):
        estimatedFinishTime = currentTime + self.remainProcessTime()
        return max(0, estimatedFinishTime - self.due_date)

    def actualTardiness(self, currentTime):
        return max(0, currentTime - self.due_date)

    def lastCompletionTime(self):
        if self.currentOpID == 0:
            return 0
        elif self.currentOpID == -1:
            return self.operations[-1].startTime + self.operations[-1].RPT
        else:
            return self.operations[self.currentOpID - 1].startTime + self.operations[self.currentOpID - 1].RPT

    def completionRate_time(self):
        return self.FPT / self.TPT

    def currentOP_percentage(self):
        if self.currentOpID == -1:
            return 0
        else:
            return self.currentOP().meanEPT / self.TPT
    
    def predictedLateness(self, global_time):
        if self.due_date <= global_time:
            return 1.0
        else: # may modify
            return max(1.0, (self.TPT - self.FPT) / (self.due_date - global_time))

    def nextOP(self):
        if self.currentOpID + 1 < len(self.operations):
            self.FPT += self.currentOP().RPT # update finished process time
            self.currentOpID += 1
            return self.currentOpID
        else:
            self.currentOpID = -1
            return -1

    def reset(self):
        self.currentOpID = 0
        self.FPT = 0
        # self.operations = [Operation(self.job_id, self.arrival_time, config) for config in self.config]
        for op in self.operations:
            op.reset(self.arrival_time)

    def isDone(self):
        return True if self.currentOpID == -1 else False
    
    def arriveAt(self, time):
        return True if self.arrival_time <= time else False

    def __len__(self):
        return len(self.operations)
    
    def show(self):
        print("%4d|%7.2f|%7.2f|"%(self.job_id, self.arrival_time, self.due_date), end='')
        for idx, op in enumerate(self.operations):
            if idx > 0:
                print("%20s|" %("") ,end = '')
            for mtime in op.EPT:
                if mtime == -1:
                    print("%7s|"%("  ---  "), end='')
                else :
                    print("%7.2f|"%(mtime), end='')
            print()
class Operation:
    def __init__(self, job_id, initTime, op_config):
        # print(EPT)
        self.machine_set = op_config['candidate_machine']
        self.initTime = initTime
        self.startTime = -1    # the time when op processed on machine
        
        self.EPT = op_config['process_time']          # Expected Process Time for every machine, -1: unable to process
        self.meanEPT = self.__meanEPT()
        self.RPT = -1
        self.job_id = job_id
        self.op_id = op_config['id']
    
    def __lt__(self, other):
        return self.job_id < other.job_id
    
    def completionTime(self):
        return self.startTime + self.RPT

    def __meanEPT(self):
        s, cnt = 0, 0
        for t in self.EPT:
            if t == -1:
                continue
            s += t
            cnt += 1
        return s / cnt

    def reset(self, job_arrival):
        self.initTime = job_arrival
        self.RPT = -1
        self.startTime = -1

    def show(self):
        print("Job id: {}, Op id: {}".format(self.job_id, self.op_id))