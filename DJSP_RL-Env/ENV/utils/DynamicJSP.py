import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import heapq
from utils import Job, Machine
import bisect

from ENV.utils.Generator import GenOperation
from ENV.utils.PaperRule import Rule1, Rule2, Rule3, Rule4, Rule5, Rule6
from djsp_logger import DJSP_Logger

class DynamicJobShopInstance:
    def __init__(self,
            Machine_num,
            Machine_num_dist_op,                # uniform dist from this range
            Initial_job_num,
            Inserted_job_num,
            DDT_range,                          # Due Date Tardiness
            Job_operation_num_dist,             # uniform dist from this range
            Process_time_dist,                  # uniform dist from this range
            RPT_effect,
            Mean_of_arrival,
            Job_type_file = None
        ):
        self.Machine_num = Machine_num
        self.Machine_num_dist_op = Machine_num_dist_op
        self.Initial_job_num = Initial_job_num
        self.Inserted_job_num = Inserted_job_num
        self.DDT_range = DDT_range
        self.Job_op_num_dist = Job_operation_num_dist
        self.Process_time_dist = Process_time_dist
        self.RPT_effect = RPT_effect
        self.Mean_of_arrival = Mean_of_arrival

        self.Machines = [Machine(machine_id, self.RPT_effect) for machine_id in range(self.Machine_num)]
        self.arrivalTime = 0
        self.currentTime = 0
        self.TimeStamp = []
        # self.TimeStamp = dict.fromkeys([i for i in range(self.Machine_num)])
        self.Jobs = []
        self.MT = []
        self.PASS = 0
        # debug
        self.time_history = []
        self.OP_history = []
        if Job_type_file is None:
            self.generate_case()
        else:
            self.load_instance(Job_type_file)
        self.logger = DJSP_Logger()
    
    def generate_case(self):
        # initial jobs
        self.insertJobs(job_num = self.Initial_job_num, init=True)
        self.insertJobs(job_num = self.Inserted_job_num)

    def insertJobs(self, job_num, init=False, op_config=None, job_type=None):
        for offset in range(job_num):
            job_id = len(self.Jobs)
            if init is False:
                arrivalTime = np.random.exponential(scale=self.Mean_of_arrival)
                self.arrivalTime += arrivalTime
                
            self.registerTime(self.arrivalTime)
            DDT = np.random.uniform(*self.DDT_range)
            
            if op_config is None:
                op_num = np.random.randint(*self.Job_op_num_dist)
                op_config = GenOperation(op_num, self.Machine_num, self.Machine_num_dist_op, self.Process_time_dist)
            else:
                op_num = len(op_config)
            typeID = job_id if job_type is None else job_type
            self.Jobs.append(
                Job(job_id=job_id, job_type=typeID, arrival_time=self.arrivalTime, DDT=DDT, op_config=op_config)
            )

    def registerTime(self, time):
        bisect.insort(self.TimeStamp, time)

    def updateTime(self):
        self.currentTime = self.TimeStamp.pop(0)
        # print("Time: ", self.currentTime)
        self.time_history.append(self.currentTime)


    def update_currentTime(self, time):
        if time > self.currentTime:
            self.currentTime = time
        for m in self.Machines:
            m.updateBuffer(self.currentTime)

    def assign(self, job_id, machine_id):
        if job_id == -1:
            return
        op = self.Jobs[job_id].currentOP()
        op_info = {
            "currentTime":  self.currentTime,
            "process_time": op.EPT,
            "job_type":     self.Jobs[job_id].job_type,
            "op_id":        op.op_id,
            "job_id":       job_id
        }
        self.OP_history.append([op_info, machine_id])
        
        
        OP_finishedTime = self.Machines[machine_id].processOP(op_info)
        self.MT.append(OP_finishedTime)
        info = self.Machines[machine_id].history[-1]
        self.Jobs[job_id].currentOP().startTime = info['startTime']
        self.Jobs[job_id].currentOP().RPT = info['RPT']
        # add op to logger
        self.logger.add_op(self.Jobs[job_id].currentOP(), machine_id, self.Jobs[job_id].job_type)
        # print('job_id: {},\tmachine_id: {},\top_id: {},\top.startTime: {},\top.RPT: {}'.format(
        #     job_id, machine_id, op.op_id, round(op.startTime, 3), round(op.RPT, 3)))
        
        if self.Jobs[job_id].nextOP() != -1:
            self.Jobs[job_id].updateCurrentOP(initTime=OP_finishedTime)
        self.registerTime(OP_finishedTime)
        # print('self.TimeStamp:', [round(t, 3) for t in self.TimeStamp])
        # print()

    def availableJobs(self):
        # self.updateTime()
        res = []
        for job in self.Jobs:
            if job.isDone():
                continue
            if job.currentOP().initTime <= self.currentTime:
                flag = False
                for m in self.Machines:
                    if m.machineTime() <= self.currentTime and job.currentOP().EPT[m.machine_id] > 0:
                        flag = True
                        break   
                if flag is True:
                    res.append(job)
        if len(res) == 0:
            self.updateTime()
            self.PASS += 1
            return self.availableJobs()
        else:
            return res

    def applyRule(self, rule):
        job_id, machine_id = rule(self.availableJobs(), self.Machines)
        self.assign(job_id, machine_id)
        
    def scheduling(self, policy):
        while not self.All_finished():
            self.applyRule(policy)
            # print(total_ops)

    def All_finished(self):
        # print("="*50)
        flag = True
        for job in self.Jobs:
            if not job.isDone():
                flag = False
        return flag


    def Estimated_Mean_Lateness(self):
        lateness, cnt = 0, 0
        for m in self.Machines:
            if len(m.history) == 0:
                continue
            for op in m.history:
                jid = op.job_id
                # RPT = sum(self.Jobs[jid])
                lateness += op.start_time + op.RPT - self.Jobs[jid].due_date
                # process_time += estimated_process_time
                cnt += 1

        if cnt == 0:
            return 0, cnt
        else:
            return float(lateness) / float(cnt), cnt

    def restart(self, refresh=False):
        self.Machines = [Machine(machine_id, self.RPT_effect) for machine_id in range(self.Machine_num)]
        if refresh is False:
            for job in self.Jobs:
                job.reset()
            self.currentTime = 0
            self.TimeStamp = []
            
            for job in self.Jobs:
                self.registerTime(job.arrival_time)
            self.BT = self.TimeStamp
        else:
            # print("Restart, init: %d, insert: %d" % (self.Initial_job_num, self.Inserted_job_num))
            self.arrivalTime = 0
            self.currentTime = 0
            self.TimeStamp = []
            # init_cnt = np.random.randint(3, self.Job_type_num + 1)
            # inserted_cnt = np.random.randint(3, self.Job_type_num * 3)
            self.Jobs = []
            for i in range(self.Initial_job_num):
                type_id = np.random.randint(self.Job_type_num)
                self.insertJobs(1, init=True, op_config=self.JobType[type_id]['op_config'], job_type=type_id)
            for i in range(self.Inserted_job_num):
                type_id = np.random.randint(self.Job_type_num)
                self.insertJobs(1, init=False, op_config=self.JobType[type_id]['op_config'], job_type=type_id)
        # print('self.TimeStamp:', self.TimeStamp)
        self.time_history = []

    def load_instance(self, file_path):
        import json
        # load basic job type
        with open(file_path, 'r') as fp:
            dic = json.load(fp)
            self.Machine_num = dic["Machine_num"]
            self.Machines = [Machine(machine_id, self.RPT_effect) for machine_id in range(self.Machine_num)]
            self.JobType = dic["JobType"]
            self.Job_type_num = len(self.JobType)
        self.Jobs = []
        tags = []
        for job in self.JobType:
            for arrivalTime, due_date in zip(job['arrivalTime'], job['due_date']):
                tags.append([job['type'], arrivalTime, due_date])
        tags = sorted(tags, key=lambda a: a[1])
        for t in tags:
            self.Jobs.append(
                Job(job_id=len(self.Jobs), job_type=t[0],arrival_time=t[1], DDT=1, op_config=self.JobType[t[0]]['op_config'])
            )
            self.Jobs[-1].due_date = t[2]
        self.restart()

        for job in self.Jobs:
            self.logger.add_job(job)

    def Tardiness(self):
        tardiness , cnt = 0, 0
        for job in self.Jobs:
            if job.currentOpID == -1:
                completionTime = job.operations[-1].startTime + job.operations[-1].RPT
                tardiness += max(0, completionTime - job.due_date)
                # print("due: %.2f, finished: %.2f" % (job.due_date, completionTime))
                cnt += 1
            else:
                print('Not finished')
        
        return tardiness

    def actualTardinessRate(self):
        # Definition: # of actual tardy operations divided by # of uncompleted operations
        # calculate without future jobs
        N_tard, N_left = 0, 0
        for job in self.Jobs:
            if job.arrival_time > self.currentTime or job.remainProcessNum() == 0:
                continue
            N_left += job.remainProcessNum()
            if job.lastCompletionTime() > job.due_date:
                N_tard += job.remainProcessNum()
        return 0 if N_left == 0 else N_tard / N_left

    def estimatedTardinessRate(self):
        # Definition: # of estimated tardy operations divided by # of uncompleted operations
        # calculate without future jobs
        T_cur = np.mean([m.machineTime() for m in self.Machines])
        N_tard, N_left = 0, 0
        for job in self.Jobs:
            if job.arrival_time > self.currentTime or job.remainProcessNum() == 0:
                continue
            N_left += job.remainProcessNum()
            T_left = 0
            for op in job.operations[job.currentOpID:]:
                T_left += op.meanEPT
                if T_cur + T_left > job.due_date:
                    N_tard += len(job.operations) - op.op_id
                    break
        return 0 if N_left == 0 else N_tard / N_left

    def Machine_UtilizationInfo(self):
        Uk = [m.utilizationRate() for m in self.Machines]
        return np.mean(Uk), np.std(Uk)

    def Job_CompletionInfo(self):
        CRJ = []
        for job in self.Jobs:
            if job.arrival_time > self.currentTime:
                continue
            CRJ.append(job.completionRate_time())
        if len(CRJ) == 0:
            return 0, 0
        else:
            return np.mean(CRJ), np.std(CRJ)

    def OP_CompletionRate(self):
        # count the job without the future ones
        complete, total = 0, 0
        for job in self.Jobs:
            if job.arrival_time > self.currentTime:
                continue
            complete += job.completedProcessNum()
            total += len(job.operations)

        return 0 if total == 0 else complete / total

    
    def makespan(self):
        return max(m.machineTime() for m in self.Machines)

    def check_schedule(self):
        for m in self.Machines:
            if m.check_invalid() is False:
                # print("Op assigned on invalid machine")
                return {
                    "error code": 1,
                    "status": "Op assigned on invalid machine"
                }
        for job in self.Jobs:
            prev_op = job.operations[0]
            
            for op in job.operations[1:]:
                if op.startTime < prev_op.startTime + prev_op.RPT:
                    return {
                        "error code": 2,
                        "status": 'OP({}, {}) assigned before previous op done'.format(job.job_id, op.op_id)
                    }
                prev_op = op

        for job in self.Jobs:
            for op in job.operations:
                if op.startTime == -1 or op.RPT == -1:
                    return {
                        "error code": 3,
                        "status": 'OP({}, {}) is not done'.format(job.jo_id, op.op_id)
                    }
        for m in self.Machines:
            if len(m.history) == 0:
                continue
            prev_op = m.history[0]
            for op in m.history[1:]:
                if op["startTime"] < prev_op["startTime"] + prev_op["RPT"]:
                    return {
                        "error code": 4,
                        "status": 'op assigned on Machine invalid'
                    }
                prev_op = op

        
        return {
            "error code": 0,
            "status": "pass"
        }

    def show_job_before_schedule(self, filename=None):
        # cmap = plt.cm.get_cmap('hsv', len(self.Jobs) + 1)
        num_jobs = len(self.Jobs)
        cmap = plt.cm.get_cmap('Set1', self.Job_type_num + 1)
        yticks = [(i + 1) * 5 for i in range(num_jobs)]
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_ylim(0, 5 * len(self.Jobs) + 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["Job {}".format(i) for i in range(num_jobs)])
        
        ax.set_xlabel('Time')
        max_due_date = 0

        data, color, op_id = [], [], []
        # print("Machine {}".format(self.machine_id))
        for job in self.Jobs:
            # print('Job Type:', job.job_type)
            # print((job.arrival_time, job.due_date))
            data.append([(job.arrival_time, job.due_date-job.arrival_time)])
            max_due_date = max(max_due_date, job.due_date)
            color.append(cmap(job.job_type))
        ax.set_xlim(0, max_due_date+100)
        ax.set_xticks(np.arange(0, max_due_date+100, step=100))
        # print(color)
        labels = ['Job Type {}'.format(t) for t in range(self.Job_type_num)]
        for i, job in enumerate(self.Jobs):
            ax.broken_barh(
                data[i], (yticks[i] - 2, 4), 
                facecolors=color[i], edgecolor='black', 
                label=labels[job.job_type])
            labels[job.job_type] = '_nolegend_'
        plt.title('Job Arrival Time & Due Time')
        plt.grid()
        plt.legend()
        if filename is not None:
            fn_list = filename.split('/')
            plt.savefig('job_plot/before_scheduling/{}.png'.format(fn_list[-1]))
        else:
            plt.show()


    def show_gantt_chart_v1(self, filename=None):
        # cmap = plt.cm.get_cmap('hsv', len(self.Jobs) + 1)
        num_jobs = len(self.Jobs)
        cmap = plt.cm.get_cmap('Set1', self.Job_type_num + 1)
        yticks = [(i + 1) * 5 for i in range(num_jobs)]
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_ylim(0, 5 * len(self.Jobs) + 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["Job {}".format(i) for i in range(num_jobs)])
        
        ax.set_xlabel('Time')

        data, color, op_id = [], [], []
        # print("Machine {}".format(self.machine_id))
        for job in self.Jobs:
            d = []
            for op in job.operations:
                d.append((op.startTime, op.RPT))
            data.append(d)
            color.append(cmap(job.job_type))
        makespan = self.makespan()
        ax.set_xlim(0, makespan+100)
        ax.set_xticks(np.arange(0, makespan+100, step=100))
        # print(color)
        labels = ['Job Type {}'.format(t) for t in range(self.Job_type_num)]
        for i, job in enumerate(self.Jobs):
            ax.broken_barh(
                data[i], (yticks[i] - 2, 4), 
                facecolors=color[i], edgecolor='black', 
                label=labels[job.job_type])
            # ax.annotate(xy=(job.due_date, yticks[i]), text="")
            ax.plot(job.due_date, yticks[i], 'bx')
            labels[job.job_type] = '_nolegend_'
        plt.title('Job Scheduling Result')
        plt.grid()
        plt.legend()
        if filename is not None:
            fn_list = filename.split('/')
            # plt.savefig('job_plot/after_scheduling/{}_{}_v1.png'.format(fn_list[-2], fn_list[-1]))
            plt.savefig('job_plot/after_scheduling/{}_v1.png'.format(fn_list[-1]))
        else:
            plt.show()

    def show_schedule(self, filename=None):
        cmap = plt.cm.get_cmap('hsv', len(self.Jobs) + 1)
        yticks = [(i + 1) * 5 for i in range(self.Machine_num)]
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_ylim(0, 5 * self.Machine_num + 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["machine {}".format(i) for i in range(self.Machine_num)])
        ax.set_xlim(0, self.makespan())
        ax.set_xlabel('Time')

        for i, m in enumerate(self.Machines):
            data, color, op_id = m.process_history(cmap)
            ax.broken_barh(data, (yticks[i] - 2, 4), facecolors=color, edgecolor='black')
            for x, d in enumerate(data):
                ST, RPT = d
                ax.text(x=ST + RPT/2, 
                    y=yticks[i],
                    s=op_id[x], 
                    ha='center', 
                    va='center',
                    color='black',
                   )
        
        label = []
        for i in range(len(self.Jobs)):
            label.append('job {}'.format(i) if i < self.Initial_job_num else 'job {}(I)'.format(i))
        # print(label)
        plt.legend([mpatches.Patch(color=cmap(i)) for i in range(len(self.Jobs))], 
                    label,
                    bbox_to_anchor=(1.1, 1.0))
        if filename is not None:
            plt.savefig('{}.png'.format(filename))
        else:
            plt.show()

    def show(self):
        print("%4s|%7s|%7s|" % ("Job", "arrival", "due"), end='')
        for idx, m in enumerate(self.Machines):
            print("%7s|" % (" M{}".format(m.machine_id)), end = '')
        print()
        # print(len(self.Machines))
        for job in self.Jobs:
            job.show()

    def show_registedJobs(self):
        print("%4s|%9s|%9s|%9s|" % ("Job", "arrival", "due date", "finish"))
        print("-"*35)
        for job in self.Jobs:
            print("%4d|%9.2f|%9.2f｜%9.2f｜" % (
                job.job_id, job.arrival_time, job.due_date, job.lastCompletionTime()))
            
            print("-"*35)


def test_DJSP(mean_of_arrival, machine_num, inserted_job, DDT, policy):
    Test_instance = DynamicJobShopInstance(
        Machine_num = machine_num,
        Machine_num_dist_op = [1, machine_num + 1],
        Initial_job_num = 20,
        Inserted_job_num = inserted_job,
        DDT_range = [DDT, DDT],
        Job_operation_num_dist = [1, 21],
        Process_time_dist = [1, 50],
        RPT_effect = False,
        Mean_of_arrival = mean_of_arrival
    )
    # Test_instance.show()
    Test_instance.scheduling(policy=policy)
    # Test_instance.show_schedule()
    tardiness, _ = Test_instance.Tardiness()
    return tardiness

def experiment(policy, DDT, LENGTH):
    Policy = policy
    E = [200]
    M = [10, 20, 30, 40, 50]
    N = [50, 100, 200]
    for e in E:
        for m in M:
            for n in N:
                tardiness = []
                for iter in range(LENGTH):
                    tardiness.append(test_DJSP(e, m, n, DDT, Policy))
                print("E: {}, M: {:2}, N_add: {:3}, Mean: {:.2f}, std: {:.2f}".format(
                    e, m, n, np.mean(tardiness), np.std(tardiness)
                ))
            print("-"*100)

def test_restart(policy):
    Test_instance = DynamicJobShopInstance(
        Machine_num = args.machine_num,
        Machine_num_dist_op = [1, args.machine_num + 1],
        Initial_job_num = 20,
        Inserted_job_num = args.inserted_job_num,
        DDT_range = [args.DDT, args.DDT],
        Job_operation_num_dist = [1, 21],
        Process_time_dist = [1, 50],
        RPT_effect = False,
        Mean_of_arrival = args.mean_of_arrival
    )
    tardiness = []
    for _ in range(args.LENGTH):
        Test_instance.scheduling(policy=policy)
        t, cnt = Test_instance.Tardiness()
        Test_instance.restart()
        tardiness.append(t)
    print(tardiness)

if __name__ == "__main__":
    rule_dict = {
        'rule1': Rule1(),
        'rule2': Rule2(),
        'rule3': Rule3(),
        'rule4': Rule4(),
        'rule5': Rule5(),
        'rule6': Rule6(),
    }
    Test_instance = DynamicJobShopInstance(
        Machine_num = 5,
        Machine_num_dist_op = [1,5],
        Initial_job_num = 20,
        Inserted_job_num = 40,
        DDT_range = [1.5, 2.3],
        Job_operation_num_dist = [1, 21],
        Process_time_dist = [1, 50],
        RPT_effect = {
            "flag": True,
            "type": "Gaussian",
            "rework_probability": 0.1,
            "rework_percentage": 0.2
        },
        Mean_of_arrival = 50
    )
    Test_instance.load_instance('./test_instance/Case7/validate_3.json')
    Test_instance.show()
    # tardiness = []
    # for _ in range(args.LENGTH):
    Test_instance.scheduling(policy=Rule4())
    print('*** schedule done ***')
    #Test_instance.show_schedule()
    Test_instance.show_registedJobs()
    tardiness = Test_instance.Tardiness()
    #     Test_instance.restart()
    #     tardiness.append(t)
    print('Tardiness: ', tardiness)
    