# The heuristic rules in paper: Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning

import numpy as np 
from random import choice

def earliestAvailableMachine(op, Machines):
    candidate_machine, available_time = [], 1e9
        
    for mid in op.machine_set:
        if Machines[mid].machineTime() < available_time:
            candidate_machine = [mid]
            available_time = Machines[mid].machineTime()

        elif Machines[mid].machineTime() == available_time:
            candidate_machine.append(mid)

    machine_id = choice(candidate_machine)
    # machine_id = candidate_machine[0]
    return machine_id


def earliestAvailableMachine_SPT(op, Machines):
    candidate_machine, available_time = [], 1e9
        
    for mid in op.machine_set:
        if Machines[mid].machineTime() < available_time:
            candidate_machine = [mid]
            available_time = Machines[mid].machineTime()

        elif Machines[mid].machineTime() == available_time:
            candidate_machine.append(mid)

    # machine_id = choice(candidate_machine)
    machine_id = candidate_machine[np.argmin([op.EPT[x] for x in candidate_machine])]
    return machine_id

class Rule1:
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
        
    def __call__(self, Jobs, Machines):
        T_cur = np.mean([m.machineTime() for m in Machines])
        tardyJobs = []
        UCJobs = []    # uncompleted jobs
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
                if job.due_date < T_cur:
                    tardyJobs.append(job)
        if len(tardyJobs) == 0:
            Ji = np.argmin([(job.due_date - T_cur)/(job.remainProcessNum()) for job in UCJobs])
            op = UCJobs[Ji].currentOP()
        else:
            Ji = np.argmax([job.remainProcessTime() - job.due_date for job in tardyJobs])
            op = tardyJobs[Ji].currentOP() 
        
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                # print('SPT')
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id
        
class Rule2:
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        T_cur = np.mean([m.machineTime() for m in Machines])
        tardyJobs, UCJobs = [], []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
                if job.due_date < T_cur:
                    tardyJobs.append(job)

        if len(tardyJobs) == 0:
            Ji = np.argmin([(job.due_date - T_cur)/(job.remainProcessTime()) for job in UCJobs])
            op = UCJobs[Ji].currentOP()
        else:
            Ji = np.argmax([job.remainProcessTime() - job.due_date for job in tardyJobs])
            op = tardyJobs[Ji].currentOP() 

        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id

class Rule3:
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        T_cur = np.mean([m.machineTime() for m in Machines])
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        Ji = np.argmax([T_cur + job.remainProcessTime() - job.due_date for job in UCJobs])
        op = UCJobs[Ji].currentOP()
        r = np.random.random()
        if r < 0.5:
            Mi = np.argmin([Machines[mid].utilizationRate() for mid in op.machine_set])
        else:
            Mi = np.argmin([Machines[mid].Workload() for mid in op.machine_set])
        machine_id = op.machine_set[Mi]
        return op.job_id, machine_id

class Rule4:
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        
        op = choice(UCJobs).currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id

class Rule5:
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        T_cur = np.mean([m.machineTime() for m in Machines])
        tardyJobs, UCJobs = [], []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
                if job.due_date < T_cur:
                    tardyJobs.append(job)
        
        if len(tardyJobs) == 0:
            Ji = np.argmin([job.completionRate() * (job.due_date - T_cur) for job in UCJobs])
            op = UCJobs[Ji].currentOP()
        else:
            Ji = np.argmax([job.completionRate() * (T_cur + job.remainProcessTime() - job.due_date) for job in tardyJobs])
            op = tardyJobs[Ji].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id
class Rule6:
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        T_cur = np.mean([m.machineTime() for m in Machines])
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)

        Ji = np.argmax([T_cur + job.remainProcessTime() - job.due_date for job in UCJobs])
        op = UCJobs[Ji].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id
