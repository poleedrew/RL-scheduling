from utils import Job, Machine, print_dict
import numpy as np
import heapq
import bisect
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from ENV.utils.Generator import GenOperation
from ENV.utils.PaperRule import Rule1
from ENV.utils.RuleByMachine import MRule1
import json

def num_before_time(arr, time, bound=False):
    # number of time >= arr[i]
    if bound is False:
        return bisect.bisect_right(arr, time, lo = 0, hi = len(arr))
    else:
        return bisect.bisect_left(arr, time, lo = 0, hi = len(arr))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class Job_type_DJSP():
    def __init__(self, config):
        self.Machine_num = config['MachineNum']
        self.Machine_num_dist_op = [1, config['OpMachineNum'] + 1]
        self.Initial_job_num = config['InitialJobNum']
        self.Inserted_job_num = config['InsertedJobNum']
        self.DDT_range = [config['DDT'], config['DDT']+config['DDT_dt']]
        self.Job_op_num_dist = [2, config['MaxOpNum'] + 1]
        self.Process_time_dist = [10., config['MaxProcessTime']]
        self.RPT_effect = config['RPT_effect']
        self.Mean_of_arrival = config['Mean_of_Arrival']
        self.Job_type_num = config['JobTypeNum']
        # self.Fixed_type = Fixed_type
        self.JobTypeFile = config['JobTypeFile']
        

        self.Machines = [Machine(machine_id, self.RPT_effect) for machine_id in range(self.Machine_num)]
        self.arrivalTime = 0
        self.currentTime = 0
        self.TimeStamp = []
        self.time_history = []
        self.PASS = 0
        if self.JobTypeFile is None or self.JobTypeFile == "":
            self.generateJobType()
            self.genJobType = True
        else:
            print('load instance: ', self.JobTypeFile)
            self.load_instance(self.JobTypeFile) # basic type
            self.genJobType = False
        self.totalOpNum = sum(len(job['op_config']) for job in self.JobType)
        # print('op num: ', self.totalOpNum)
        self.insertJobs(self.Initial_job_num, init=True)
        self.insertJobs(self.Inserted_job_num)
        self.has_job_done = False
        self.MT = []
        # print("[init] init: %d, insert: %d" % (self.Initial_job_num, self.Inserted_job_num))

    def updateJobNum(self, initial, insert = 0):
        self.Initial_job_num = initial
        self.Inserted_job_num = insert

    def generateJobType(self):
        self.JobType = []
        for type_id in range(self.Job_type_num):
            job_config = GenOperation(
                op_num = np.random.randint(*self.Job_op_num_dist), 
                machine_num = self.Machine_num, 
                machine_num_dist_op = self.Machine_num_dist_op, 
                process_time_dist = self.Process_time_dist)
            job_info = {
                "type": type_id,
                "due_date": [],
                "arrivalTime": [],
                "job_finishedTime": [],
                "op_config": job_config,
                "op_finishedTime": [[] for _ in range(len(job_config))],
                "op_startTime": [[] for _ in range(len(job_config))],
            }
            accumulate = np.zeros(len(job_info["op_config"]))
            for op in job_info["op_config"]:
                accumulate[op["id"]] = op["MPT"] + (0 if op["id"] == 0 else accumulate[op["id"] - 1])
            accumulate = accumulate[::-1]
            job_info["aMPT"] = accumulate
            # print(job_info["aMPT"])
            self.JobType.append(job_info)

    def insertJobs(self, job_num, init=False):
        for offset in range(job_num):
            type_id = np.random.randint(self.Job_type_num) if self.genJobType is False else offset % self.Job_type_num
            if init is False:
                arrivalTime = np.random.exponential(scale=self.Mean_of_arrival)
                self.arrivalTime += arrivalTime
                
            self.registerTime(self.arrivalTime)
            DDT = np.random.uniform(*self.DDT_range)
            due_date = self.arrivalTime + DDT * sum(config["MPT"] for config in self.JobType[type_id]["op_config"])
            bisect.insort(self.JobType[type_id]["due_date"], due_date)
            bisect.insort(self.JobType[type_id]["arrivalTime"], self.arrivalTime)

    def estimatedTardiness(self, typeidx, jobidx):
        jobTime = 0
        opNum = len(self.JobType[typeidx]['op_config'])
        for i in range(opNum):
            if jobidx < len(self.JobType[typeidx]['op_finishedTime'][i]):
                jobTime = max(jobTime, self.JobType[typeidx]['op_finishedTime'][i][jobidx])
            else:
                estimatedFinishTime = max(jobTime, self.currentTime) + self.JobType[typeidx]['aMPT'][i]
                
                return max(0, estimatedFinishTime - self.JobType[typeidx]['due_date'][jobidx])
        
        return -1


    def refresh_arrival(self):
        # init_cnt = self.Initial_job_num
        # inserted_cnt = self.Initial_job_num
        # print("refresh, init: %d, insert: %d" % (init_cnt, inserted_cnt))
        for job in self.JobType:
            job["arrivalTime"] = []
            job["due_date"] = []
        self.Fixed_type = False
        self.insertJobs(self.Initial_job_num, init=True)
        self.insertJobs(self.Inserted_job_num, init=False)

    def registerTime(self, time):
        bisect.insort(self.TimeStamp, time)

    def updateTime(self):
        self.currentTime = self.TimeStamp.pop(0)
        # print("Time: ", self.currentTime)
        self.time_history.append(self.currentTime)

    def assign(self, op, machine_id):
        # print("Assign O({},{}) on machine {}".format(op["job_type"], op["op_id"], machine_id))
        self.has_job_done = False
        if op == -1:
            # noop
            time_idx = bisect.bisect_right(self.TimeStamp, self.currentTime, lo = 0, hi = len(self.TimeStamp))
            if time_idx < len(self.TimeStamp):
                self.registerTime(self.TimeStamp[time_idx])
            return

        op["currentTime"] = self.currentTime
        finishedTime = self.Machines[machine_id].processOP(op)
        info = self.Machines[machine_id].history[-1]
        self.MT.append(finishedTime)
        bisect.insort(self.JobType[op["job_type"]]["op_finishedTime"][op["op_id"]], finishedTime)
        bisect.insort(self.JobType[op["job_type"]]["op_startTime"][op["op_id"]], info['startTime'])
        if self.__is_last_op(op):
            bisect.insort(self.JobType[op["job_type"]]["job_finishedTime"], finishedTime)
            # print("one job finished at ", finishedTime)
            self.prev_finishedJob = op["job_type"]
            self.has_job_done = True
            # self.show_registedJobs()
        self.registerTime(finishedTime)

    def available_job_info(self):
        self.updateTime()
        
        res = []
        machine_mask = self.machine_mask()
        for job in self.JobType:
            for op in job["op_config"]:
                machine_process_time = op["process_time"] * machine_mask
                if all(x <= 0 for x in machine_process_time):
                    res.append(-1)
                elif self.__available_op(op, job):
                    res.append({
                        "job_type": job["type"],
                        "op_id": op["id"],
                        "process_time": machine_process_time
                    }) 
                else:
                    res.append(-1)
        # tmp_mask = [0 if x == -1 else 1 for x in res]
        # print("tmp_mask: ", tmp_mask)
        if all(x == -1 for x in res):
            self.PASS += 1
            return self.available_job_info()
        else:
            return res

    def __available_op(self, op, job):
        op_start_num = len(job["op_startTime"][op["id"]])
        op_finished_num = num_before_time(job["op_finishedTime"][op["id"]], self.currentTime)
        job_num = num_before_time(job["arrivalTime"], self.currentTime)
        # print("Job {}, job_num: {}; op {}, started: {},finished: {}".format(job["type"], job_num, op["id"],op_start_num, op_finished_num))
        # if op['id'] > 0:
        #     print(num_before_time(job["op_finishedTime"][op["id"] - 1], self.currentTime))
        if (op["id"] == 0 and op_start_num < job_num):
            return True
        elif (op["id"] > 0 and op_start_num < num_before_time(job["op_finishedTime"][op["id"] - 1], self.currentTime)):
            # print("found")
            return True
        else:
            return False

    def applyRule(self, rule):

        op_info = self.available_job_info()
        op, machine_id = rule(op_info, self.Machines)
        #     # print(action)
        self.assign(op, machine_id)

    def scheduling(self, policy):
        self.noop_cnt = 4
        while not self.All_finished():
            self.applyRule(policy)

    def All_finished(self):
        
        if all(len(job["job_finishedTime"]) == len(job["arrivalTime"]) for job in self.JobType):
            return True
        elif len(self.TimeStamp) == 0:
            # print("too much noop")
            return True
        else :
            return False

    def machine_mask(self):
        machine_mask = np.zeros_like(self.Machines)
        for m in self.Machines:
            if self.currentTime >= m.machineTime():
                machine_mask[m.machine_id] = 1
        return machine_mask

    def __is_last_op(self, op):
        if op["op_id"] == len(self.JobType[op["job_type"]]["op_config"]) - 1:
            return True
        else:
            return False

    def restart(self, refresh=False):
        self.arrivalTime = 0
        self.currentTime = 0
        self.TimeStamp = []
        self.time_history = []
        self.has_job_done = False
        self.prev_finishedJob = -1
        for m in self.Machines:
            m.reset()
        
        for job in self.JobType:
            job["job_finishedTime"] = []
            job["op_finishedTime"] = [[] for _ in range(len(job["op_config"]))]
            job["op_startTime"] = [[] for _ in range(len(job["op_config"]))]
            # for arrivalTime in job["arrivalTime"]:
            #     self.registerTime(arrivalTime)

        if refresh is True:
            self.genJobType = False
            self.refresh_arrival()
        else:
            for job in self.JobType:
                for arrivalTime in job["arrivalTime"]:
                    self.registerTime(arrivalTime)
            self.BT = self.TimeStamp

    def show_schedule(self, filename=None):
        cmap = plt.cm.get_cmap('hsv', self.Job_type_num + 1)
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
        for i in range(self.Job_type_num):
            label.append('Job type {}'.format(i))
        # print(label)
        plt.legend([mpatches.Patch(color=cmap(i)) for i in range(self.Job_type_num)], 
                    label,
                    bbox_to_anchor=(1.1, 1.0))
        if filename is not None:
            plt.savefig('{}.png'.format(filename))
        else:
            plt.show()

    def makespan(self):
        return max(m.machineTime() for m in self.Machines)

    def Tardiness(self):
        tardiness = 0
        for job in self.JobType:
            for due_date, finishedTime in zip(job["due_date"], job["job_finishedTime"]):
                tardiness += max(0, finishedTime - due_date)
        return tardiness

    def show(self):
        print("%4s|" % ("Job"), end='')
        for idx, m in enumerate(self.Machines):
            print("%7s|" % (" M{}".format(m.machine_id)), end = '')
        print("%7s"%("MPT"))
        print("-"*(5 + 8 * self.Machine_num + 8))
        # print(len(self.Machines))
        for job in self.JobType:
            self.show_job_type(job)
            print("-"*(5 + 8 * self.Machine_num))

    def show_job_type(self, job):
        # print("job has %d op" % (len(job["op_config"])))
        print("%4d|"%(job["type"]), end='')
        for op in job["op_config"]:
            if op["id"] > 0:
                print("%4s|" %("") ,end = '')
            for mtime in op["process_time"]:
                if mtime == -1:
                    print("%7s|"%("  ---  "), end='')
                else :
                    print("%7.2f|"%(mtime), end='')
            print("%7.2f|" % (op["MPT"]))

    def show_registedJobs(self):
        print("%4s|%9s|%9s|%9s|" % ("Job", "arrival", "due date", "finish"))
        print("-"*35)
        for job in self.JobType:
            print("%4d|" % (job["type"]), end = '')
            if len(job["job_finishedTime"]) == 0:
                job_finishedTime = [-1] * len(job["arrivalTime"])
            else:
                job_finishedTime = job["job_finishedTime"]
            for idx, (arrival, due_date, finishedTime) in enumerate(zip(job["arrivalTime"], job["due_date"], job_finishedTime)):
                if idx > 0:
                    print("%4s|" % (""), end='')
                print("%9.2f|%9.2f｜%9.2f｜" % (arrival, due_date, finishedTime))
            if len(job["arrivalTime"]) == 0:
                print()
            print("-"*35)

    def check_schedule(self):
        for m in self.Machines:
            if m.check_invalid() is False:
                # print("Op assigned on invalid machine")
                return {
                    "error code": 1,
                    "status": "Op assigned on invalid machine"
                }
        
        # for job in self.JobType:
            

        for job in self.JobType:
            for idx, startTime in enumerate(job["op_startTime"][0]):
                # first op
                num = 0
                for arrivalTime in job['arrivalTime']:
                    if arrivalTime <= startTime:
                        num += 1
                    else:
                        break

                if num < idx + 1:
                    print(job['op_startTime'][0])
                    print(job['arrivalTime'])
                    return {
                        "error code": 2,
                        "status": "Job type {}, no.{}, OP({}) starts before job arrives, start: {}, job arrives: {}".format(job["type"],idx ,  0, startTime, job['arrivalTime'][num])
                    }

            for i in range(len(job['arrivalTime'])):
                for op_id in range(len(job['op_config']) - 1):
                    if job['op_finishedTime'][op_id][i] > job['op_startTime'][op_id + 1][i]:
                        print('job type: ', job['type'])
                        print('op id: ', op_id)
                        print(job['op_finishedTime'][op_id])
                        print(job['op_startTime'][op_id + 1])
                        return {
                            'error code': 3,
                            'status': 'op({}, {}) assigned before previous end'.format(job['type'], op_id + 1)
                        }

            if len(job["job_finishedTime"]) != len(job["arrivalTime"]):
                # print("job {} are not finished".format(job["type"]))
                return {
                    "error code": 4,
                    "status": "job {} are not finished".format(job["type"])
                }

        for i in range(1, len(self.time_history)):
            if self.time_history[i] < self.time_history[i - 1]:
                print("time history wrong")
                return {
                    "error code": 5,
                    "status": "time history is wrong"
                }

        return {
            "error code": 0,
            "status": "finish scheduling"
        }

    def save_instance(self, file_path):
        import json
        dic = {
            "Machine_num": self.Machine_num,
        }
        job_dict = []
        for job in self.JobType:
            info = {
                "type": job["type"],
                "arrivalTime": job["arrivalTime"],
                "due_date": job["due_date"],
                "aMPT": job["aMPT"],
                "op_config": []
            }
            for op in job["op_config"]:
                op_dict = {
                    "id": op["id"],
                    "candidate_machine": op["candidate_machine"],
                    "process_time": op["process_time"],
                    "MPT": op["MPT"]
                }
                info["op_config"].append(op_dict)
            
            job_dict.append(info)
        dic["JobType"] = job_dict
        dic["DDT_range"] = self.DDT_range
        # print(dic)
        json = json.dumps(dic, cls=NpEncoder)
        with open(file_path, 'w') as fp:
            fp.write(json)

    def load_instance(self, file_path):
        with open(file_path, 'r') as fp:
            dic = json.load(fp)
            self.Machine_num = dic["Machine_num"]
            self.Machines = [Machine(machine_id, self.RPT_effect) for machine_id in range(self.Machine_num)]
            self.JobType = dic["JobType"]
            self.Job_type_num = len(self.JobType)
        # self.show_registedJobs()
        for jobtype in self.JobType:
            tmp, accumulateMPT = 0, []
            for op in reversed(jobtype['op_config']):
                tmp += op['MPT']
                accumulateMPT.insert(0, tmp)
            self.JobType[jobtype['type']]['aMPT'] = accumulateMPT

        for jobtype in self.JobType: 
            print(jobtype["aMPT"])
        self.restart()

def Varify_instance():
    from tqdm import tqdm
    for _ in tqdm(range(100000)):
        Test_instance = Job_type_DJSP(
            Machine_num = args.machine_num,
            Machine_num_dist_op = [1,args.machine_num ],
            Initial_job_num = args.initial_job_num,
            Inserted_job_num = args.inserted_job_num,
            DDT_range = [args.DDT, args.DDT],
            Job_operation_num_dist = [1, 5],
            Process_time_dist = [20, 50],
            RPT_effect = args.RPT_effect,
            Mean_of_arrival = args.mean_of_arrival,
            Job_type_num = args.job_type_num
        )
        # Test_instance.show()
        # Test_instance.show_registedJobs()
        Test_instance.scheduling(policy=MRule1())
        if Test_instance.check_schedule() is False:
            print("Incorrect schedule result")
            Test_instance.show_registedJobs()
            Test_instance.show_schedule()
            
            break



if __name__ == '__main__':
    # Varify_instance()
    
    Test_instance = Job_type_DJSP(
        Machine_num = args.machine_num,
        Machine_num_dist_op = [1,args.machine_num ],
       Initial_job_num = args.initial_job_num,
        Inserted_job_num = args.inserted_job_num,
        DDT_range = [args.DDT, args.DDT],
        Job_operation_num_dist = [1, 5],
        Process_time_dist = [1, 50],
        RPT_effect = args.RPT_effect,
        Mean_of_arrival = args.mean_of_arrival,
        Job_type_num = args.job_type_num
    )
    
    Test_instance.load_instance(args.djsp_instance)
    Test_instance.show_registedJobs()
    Test_instance.restart(refresh=False)
    Test_instance.show_registedJobs()
    Test_instance.restart(refresh=True)
    Test_instance.show_registedJobs()
    # Test_instance.scheduling(policy=MRule1())
    # print("noop: {}".format(4 - Test_instance.noop_cnt))
    # Test_instance.show_registedJobs()
    # Test_instance.show_schedule()

       