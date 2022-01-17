import numpy as np 
from random import choice
from ENV.utils.PaperRule import earliestAvailableMachine, earliestAvailableMachine_SPT

class EDD:
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        # If many have same arrival time, select earliest due date
        UCJobs.sort(key=lambda j: j.due_date)
        op = UCJobs[0].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id


class FIFO:
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        # If many have same arrival time, select earliest due date
        UCJobs.sort(key=lambda j: (j.arrival_time, j.due_date))
        op = UCJobs[0].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id

class SPT:
    # Shortest process time(total)
    # If many, select earliest due date
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        UCJobs.sort(key=lambda job: (job.TPT, job.due_date))
        op = UCJobs[0].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id

class LPT:
    # Longest process time
    # If many, select earliest due date
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        UCJobs.sort(key=lambda job: (job.TPT*-1, job.due_date))
        op = UCJobs[0].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id
class SRPT:
    # Shortest remaining process time
    # If many, select earliest due date
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        UCJobs.sort(key=lambda job: (job.remainProcessTime(), job.due_date))
        op = UCJobs[0].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id

class LS:
    # Least Slack
    # s(j) = due date - remaining process time
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        UCJobs.sort(key=lambda job: (job.due_date - job.remainProcessTime()))
        op = UCJobs[0].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id

class CR:
    # critical ratio
    # A ratio less than 1.0 implies that the job is behind schedule, 
    # and a ratio greater than 1.0 implies that the job is ahead of schedule. 
    # The job with the lowest CR is scheduled next.
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        T_cur = np.mean([m.machineTime() for m in Machines])
        UCJobs.sort(key=lambda job: ((job.due_date - T_cur)/(job.remainProcessTime()), job.due_date))
        op = UCJobs[0].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id

class MOR:
    # Most Operation Remaining
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        UCJobs.sort(key=lambda job: (len(job.operations)-job.currentOpID-1, job.due_date), reverse=True)
        op = UCJobs[0].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id

class LOR:
    # Least Operation Remaining
    def __init__(self, rule_config=None):
        self.rule_config = rule_config
    def __call__(self, Jobs, Machines):
        UCJobs = []
        for job in Jobs:
            if job.currentOpID != -1:
                UCJobs.append(job)
        UCJobs.sort(key=lambda job: (len(job.operations)-job.currentOpID-1, job.due_date))
        op = UCJobs[0].currentOP()
        if self.rule_config is None or 'machineSelection' not in self.rule_config:
            machine_id = earliestAvailableMachine(op, Machines)
        else:
            if self.rule_config['machineSelection'] == 'SPT': # shortest process time
                machine_id = earliestAvailableMachine_SPT(op, Machines)
            else:
                machine_id = earliestAvailableMachine(op, Machines)
        # select earlist available machine
        return op.job_id, machine_id
