import gym
from gym.utils import EzPickle
from ENV.utils.Job_type_DJSP import Job_type_DJSP, num_before_time
import numpy as np 
from gym import spaces
from random import choice
import os
from utils import json_to_dict
from gym.spaces import Box, Dict, Discrete
def random_pick(mask):
    candidate = []
    for i, val in enumerate(mask):
        if val == 1:
            candidate.append(i)
    return choice(candidate)
class Foo_Env2(gym.Env, EzPickle):
    def __init__(self, env_config):
        self.action_space = spaces.Discrete(5)
        # self.observation_space = spaces.Box(low=np.float32(0.), high=np.float32(255.), dtype=np.float32, shape=(125,))
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(5, )),
            "avail_actions": Box(0, 1, shape=(5, 1)),
            "observation": spaces.Box(low=np.float32(0.), high=np.float32(255.), dtype=np.float32, shape=(125,))
        })
        # print(self.observation_space)
        self.cnt = 0
        self.avail = [[1] for _ in range(5)]
        self.avail[3][0] = 1 
        self.mask = np.array([1, 1, 1, 1, 1])
        self.limit = 4
        self.tag = ''

    def step(self, action):
        self.cnt += 1
        if action == 3:
            reward = -10
            done = True
        elif self.cnt >= self.limit:
            reward = 1
            done = True
        else:
            done = False
            reward = 1
        
        self.mask[action] = 0

        if self.tag != '':
            info = {self.tag: self.limit}
        else:
            info = {}
        return self.current_state(), reward, done, info

    def reset(self):
        self.cnt = 0
        self.avail = [[1] for _ in range(5)]
        self.avail[3][0] = 1
        self.mask = np.array([1, 1, 1, 1, 1])
        # self.tag = ''
        return self.current_state()

    def current_state(self):
        state_dict = {
            'observation': np.random.rand(125),
            'action_mask': self.mask,
            'avail_actions': np.array(self.avail)
        }
        return state_dict
    
    def set_eval(self, length):
        self.tag = 'tag_{}'.format(length)
        self.limit = length

    def foo_info(self):
        return self.limit + 100

class DJSP_Env2(gym.Env, EzPickle):
    def __init__(self, env_config):
        
        self.djspArgs = json_to_dict(env_config['djspArgsFile'])
        print(self.djspArgs)
        self.DJSP_Instance = Job_type_DJSP(self.djspArgs)
        # self.load_basic_type()
        # self.DJSP_Instance.show()
        self.__fixed_case = False if 'fixed_case' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['fixed_case']
        self.noop = False if 'noop' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['noop']
        self.maxJobTypeNum = 5 if 'maxJobTypeNum' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['maxJobTypeNum']
        self.maxJobTypeNum = 5 if 'maxJobTypeNum' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['maxJobTypeNum']
        self.maxOPperJob = 5 if 'maxOPperJob' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['maxOPperJob'] 
        self.maxMachineNum = 5 if 'maxMachineNum' not in self.djspArgs['ENV'] else self.djspArgs['ENV']['maxMachineNum'] 
        self.__init_space()
        self.evaluation_flag = False if 'testing' not in env_config else env_config['testing']
        self.evaluation_file = "" if self.evaluation_flag is False else self.djspArgs['JobTypeFile']
        print('evaluation: ', self.evaluation_flag)
        print('fixed case: ', self.__fixed_case)
        self.prev_estimatedTardiness = 0
        self.result_index = 0
        self.reset()
    
    def __new_djsp_instance(self):
        self.DJSP_Instance = Job_type_DJSP(
            Machine_num = self.config["Machine_num"],
            Machine_num_dist_op = self.config["Machine_num_dist_op"],
            Initial_job_num=self.config["Initial_job_num"],
            Inserted_job_num=self.config["Inserted_job_num"],
            DDT_range=self.config["DDT_range"],
            Job_operation_num_dist=self.config["Job_operation_num_dist"],
            Process_time_dist=self.config["Process_time_dist"],
            RPT_effect=self.config["RPT_effect"],
            Mean_of_arrival=self.config["Mean_of_arrival"],
            Job_type_num=self.config["Job_type_num"],
            Fixed_type=self.config["Fixed_type"]
        )

    def __init_space(self):
        self.__define_action()
        self.__preprocess_state()
        self.__finished_job = np.zeros(self.DJSP_Instance.Job_type_num)
        self.noop_cnt = 0
        # self.update_state()
        # s = self.current_state()
        # self.action_space = spaces.Discrete(len(self.action))
        # self.observation_space = spaces.Box(low=np.float32(-128.), high=np.float32(128.), dtype=np.float32, shape=s.shape)
        obs_space_n = self.DJSP_Instance.totalOpNum * self.DJSP_Instance.Machine_num * 2 +\
                        self.DJSP_Instance.Machine_num +\
                        self.DJSP_Instance.totalOpNum +\
                        self.DJSP_Instance.Job_type_num * 2 * 2 + \
                        (1 if self.noop is True else 0)
        print('obs_space_n: ', obs_space_n)
        self.avail_actions = [[1.] for _ in range(self.action_space.n)]
        if self.noop is True:
            self.avail_actions.append([1.])
        self.avail_actions = np.array(self.avail_actions)
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.action_space.n, )),
            "avail_actions": Box(0, 1, shape=(self.action_space.n, 1)),
            "observation": spaces.Box(low=np.float32(-128.), high=np.float32(128.), dtype=np.float32, shape=
                (obs_space_n,))
        })
        # print(self.observation_space)
        #print(s.shape)

    def reset(self, refresh=True):
        # self.prevState = np.array([0. for _ in self.DJSP_Instance.Machine_num])
        # self.DJSP_Instance.show()
        # self.DJSP_Instance.show_registedJobs()
        if self.evaluation_flag is True:
            self.DJSP_Instance.restart(refresh=False)
        elif self.__fixed_case is True:
            self.DJSP_Instance.restart(refresh=False)
        else:
            self.DJSP_Instance.restart(refresh=refresh)
        self.noop_cnt = 0
        self.prev_estimatedTardiness = 0
        self.prev_actualTardiness = 0
        self.prev_machineUtilization = 0
        
        # self.DJSP_Instance.show()
        # self.DJSP_Instance.show_registedJobs()
        self.update_state()
        return self.current_state()

    def restart(self, refresh=False):
        # restart the same DJSP instance
        self.DJSP_Instance.restart(refresh=refresh)
        self.noop_cnt = 0
        # self.DJSP_Instance.show()
        # self.DJSP_Instance.show_registedJobs()
        self.update_state()
        return self.current_state()

    def step(self, action_id):
        # action_id = action_id[0]
        if self.action[action_id]["job_type"] == "noop":
            # print("hi")
            self.noop_cnt += 1
            self.DJSP_Instance.assign(-1, 0)
            reward, done, info = self.reward_function(action_id), self.DJSP_Instance.All_finished(), {"note": "noop"}
            if not done:
                self.update_state()

        elif self.action_mask[action_id] == -1:
            reward, done, info = -10, True, {"error": "invalid action"}
            exit()

        else:
            machine_id = self.__maskAction2machine(action_id)
            self.DJSP_Instance.assign(self.action[action_id], machine_id)
            reward, done, info = self.reward_function(action_id), self.DJSP_Instance.All_finished(), {}
            # self.machine_cnt -= self.op_info[action]["process_time"] > 0
            if not done:
                self.update_state()

        # if done is True and self.DJSP_Instance.check_schedule()["error code"] == 4:
        #     reward += -5

        # info["machine_id"] = self.assign_machine
        # if done is True and self.evaluation_flag is True:
        if done is True:
            info[self.evaluation_file] = self.DJSP_Instance.Tardiness()
            # info['{}_check'.format(self.evaluation_file)] = self.DJSP_Instance.check_schedule()['error code']
            if self.evaluation_flag is False:
                self.DJSP_Instance.show_schedule(filename='./gant/ep_{}.png'.format(self.result_index))
            else:
                self.DJSP_Instance.show_schedule(filename='./gant/eval_ep_{}.png'.format(self.result_index))
            self.result_index += 1
        
        if self.__fixed_case is True:
            info['fixed_case'] = 1

        return self.current_state(), reward, done, info

    def update_state(self):
        # DJSP_Instance.available_job_info() has masked out process time in op
        self.op_info = self.DJSP_Instance.available_job_info()
        machine_mask = self.DJSP_Instance.machine_mask()
        self.action_mask = np.zeros_like(self.originActionMask)
        for op in self.op_info:
            if op == -1:
                continue
            for machine_id, process_time in enumerate(op['process_time']):
                if process_time > 0:
                    job_offset = self.typeMap[op['job_type']]
                    action_id = self.__action_id(job_offset, op['op_id'], machine_id)
                    self.action_mask[action_id] = 1

        # self.machine_cnt = np.zeros_like(self.DJSP_Instance.Machines)
        
        # for idx, op in enumerate(self.op_info):
        #     if op == -1:
        #         continue
        #     self.machine_cnt += op["process_time"] > 0
        # self.machine_cnt *= machine_mask
        # self.machine_set = []
        # for i, cnt in enumerate(self.machine_cnt):
        #     if cnt > 0:
        #         self.machine_set.append(i)
        # self.assign_machine = choice(self.machine_set)
        # self.action_mask = np.zeros(self.action_space.n)
        # for idx, op in enumerate(self.op_info):
        #     if op != -1 and op["process_time"][self.assign_machine] > 0:
        #         self.action_mask[idx] = 1
        if self.noop is True:
            self.action_mask[-1] = 1


    def current_state(self):
        op_mask = np.array([int(not x == -1) for x in self.op_info])
        machine_mask = np.zeros(self.DJSP_Instance.Machine_num)
        # machine_mask[self.assign_machine] = 1
        job_info, tardy_info = self.__job_info()
        s = np.concatenate((self.OP_table, op_mask, machine_mask, job_info, tardy_info), axis=0)
        # s = np.concatenate((op_mask, machine_mask, job_info, tardy_info), axis=0)
        
        # self.observation_space = [self.OP_table.shape, len(s)]
        #print(s.shape)
        state = {
            'observation': s,
            'action_mask': self.action_mask,
            'avail_actions': self.avail_actions
        }
        return state

    def __job_info(self):
        job_info = []
        # job finished rate
        for job in self.DJSP_Instance.JobType:
            # non-finished job count
            job_cnt = num_before_time(job["arrivalTime"], self.DJSP_Instance.currentTime) - len(job["job_finishedTime"])
            # print("job_cnt: ", job_cnt)
            if job_cnt == 0:
                job_info += [1.0, 0.]
            else:
                finishRate = []
                for idx in range(len(job["job_finishedTime"]), len(job["job_finishedTime"]) + job_cnt):
                    process_time, denominator = 0., 0.
                    for opid in range(len(job['op_config'])):
                        if len(job['op_startTime'][opid]) <= idx:
                            denominator += job['op_config'][opid]['MPT']
                        else:
                            denominator += (job['op_finishedTime'][opid][idx] - job['op_startTime'][opid][idx])
                            process_time += min(job['op_finishedTime'][opid][idx],self.DJSP_Instance.currentTime) - job['op_startTime'][opid][idx]
                    finishRate.append(process_time/denominator)
                job_info += [np.mean(finishRate), np.std(finishRate)]
        
        # job delay rate, estimated 
        tardy_info = []
        normalize = np.vectorize(lambda x, y: x/(-x + y) if x < 0 else x / (x+y))
        flash_time = np.vectorize(lambda x, y: max(x, y))
        for job in self.DJSP_Instance.JobType:
            job_cnt = num_before_time(job["arrivalTime"], self.DJSP_Instance.currentTime)
            if job_cnt == 0 or job_cnt == len(job["job_finishedTime"]):
                tardy_info += [0., 0.]
                continue
            estimated_endTime = np.ones(job_cnt - len(job["job_finishedTime"])) * self.DJSP_Instance.currentTime + job["aMPT"][0]
            startID = len(job["job_finishedTime"])
            for op in job["op_config"][:-1]:
                fop_cnt = len(job["op_finishedTime"][op["id"]]) - len(job["job_finishedTime"])
                if fop_cnt <= 0:
                    continue
                estimated_endTime[0:fop_cnt] = flash_time(job["op_finishedTime"][op["id"]][startID:], self.DJSP_Instance.currentTime) + np.ones(fop_cnt)*job["aMPT"][op["id"] + 1]

            dif = estimated_endTime - job["due_date"][startID:startID + len(estimated_endTime)] 
            dif = normalize(dif, job["aMPT"][-1])
            tardy_info += [np.mean(dif), np.std(dif)]

        return np.array(job_info), np.array(tardy_info)

    def __preprocess_state(self):
        denominator = self.DJSP_Instance.Process_time_dist[1]
        normalize = np.vectorize(lambda x, y: x if x < 0 else x / y)
        self.OP_timetable = None
        self.OP_percentage = None
        for job in self.DJSP_Instance.JobType:
            job_denominator = sum(op["MPT"] for op in job["op_config"])
            # print(job_denominator)
            for op in job["op_config"]:
                timetable = normalize(np.expand_dims(op["process_time"], axis=0), denominator)
                percentage = normalize(np.expand_dims(op["process_time"], axis=0), job_denominator)
                if self.OP_timetable is None:
                    self.OP_timetable, self.OP_percentage = timetable, percentage
                else:
                    self.OP_timetable = np.concatenate((self.OP_timetable, timetable), axis = 0)
                    self.OP_percentage = np.concatenate((self.OP_percentage, percentage), axis=0)
        # print(self.OP_percentage.flatten().shape)
        # print(self.OP_timetable.flatten().shape)
        # self.OP_table = np.concatenate((np.expand_dims(self.OP_timetable, axis=0), np.expand_dims(self.OP_percentage, axis=0)), axis = 0)
        self.OP_table = np.concatenate((self.OP_percentage.flatten(), 
                                        self.OP_timetable.flatten()), axis=0)
        # new state
        # each operation has 5 feature
        # 1. available
        # 2. 
        self.originStateMask = np.zeros((self.maxJobTypeNum * self.maxOPperJob, 5))


    def reward_function(self, action_id):
        res = -0.2 if self.action[action_id]["job_type"] == "noop" else 0
        denseReward = self.denseReward()
        if self.DJSP_Instance.has_job_done is True:
            doneReward = self.__job_done_inTime()
        else:
            doneReward = 0
        # print('doneR: {:.2f}, denseR: {:.2f}'.format(doneReward, denseReward))
        return denseReward/10


    def isDone(self):
        return self.DJSP_Instance.All_finished()
    
    def denseReward(self):
        R = 0
        machineUtilization = np.mean([m.utilizationRate() for m in self.DJSP_Instance.Machines])
        # R = machineUtilization - self.prev_machineUtilization
        
        total_estimatedTardiness = 0
        total_actualTardiness = 0
        for typeidx in range(len(self.DJSP_Instance.JobType)):
            jobtype = self.DJSP_Instance.JobType[typeidx]
            jobidx = len(jobtype['job_finishedTime']) # non-finished job
            while jobidx < len(jobtype['arrivalTime']) and\
                jobtype['arrivalTime'][jobidx] <= self.DJSP_Instance.currentTime:
                # exist jobs
                # estimated tardiness
                estimatedTardiness = self.DJSP_Instance.estimatedTardiness(typeidx, jobidx)
                total_estimatedTardiness += estimatedTardiness
                total_actualTardiness += max(0, self.DJSP_Instance.currentTime - jobtype['due_date'][jobidx])
                if estimatedTardiness < 0:
                    print('[Error] estimatedTardiness in JobType DJSP')
                    exit()
                jobidx += 1
        # calculate reward 
        if total_actualTardiness < self.prev_actualTardiness:
            R = 1.
        else:
            if total_actualTardiness > self.prev_actualTardiness:
                R = -1.
            else:
                if total_estimatedTardiness < self.prev_estimatedTardiness:
                    R = 1.
                else:
                    if total_estimatedTardiness > self.prev_estimatedTardiness:
                        R = -1.
                    else:
                        if machineUtilization > self.prev_machineUtilization:
                            R = 1.
                        elif machineUtilization < self.prev_machineUtilization * 0.95:
                            R = 0.
                        else:
                            R = -1
        self.prev_machineUtilization = machineUtilization
        self.prev_estimatedTardiness = total_estimatedTardiness
        self.prev_actualTardiness = total_actualTardiness
        return R


    def __job_done_inTime(self):
        jobID = self.DJSP_Instance.prev_finishedJob
        # if jobID == -1:
        #     print("xxxxx")
        #     return 0
        idx = len(self.DJSP_Instance.JobType[jobID]["job_finishedTime"]) - 1
        # print("jobid: {}, idx: {}".format(jobID, idx))
        finishedTime = self.DJSP_Instance.JobType[jobID]["job_finishedTime"][-1]
        due_date = self.DJSP_Instance.JobType[jobID]["due_date"][idx]
        res = 1.0 if finishedTime <= due_date else (due_date - finishedTime)/self.reward_denominator
        return res

    def __define_action(self):
        self.action_space = spaces.Discrete(self.maxJobTypeNum * self.maxOPperJob * self.maxMachineNum + (1 if self.noop is True else 0))
        self.action = [{} for _ in range (self.action_space.n)]
        self.originActionMask = np.zeros(self.action_space.n)
        self.reward_denominator = 0
        self.typeMap = {}
        assert self.maxJobTypeNum >= len(self.DJSP_Instance.JobType), '# of job type too many.'
        for job_offset, job in enumerate(self.DJSP_Instance.JobType):
            self.typeMap[job['type']] = job_offset
            assert self.maxOPperJob >= len(job['op_config']), '# of operation in job type {} too many'.format(job['type'])
            for op_offset, op in enumerate(job["op_config"]):
                assert self.maxMachineNum >= len(op['process_time']), '# of machine is too many'
                for machine_offset, process_time in enumerate(op['process_time']):
                    if process_time <= 0:
                        continue
                    action_id = self.__action_id(job_offset, op_offset, machine_offset)
                    # print('a id: ', action_id)
                    self.action[action_id] = {
                        "job_type": job["type"],
                        "op_id": op["id"],
                        'machine_id': machine_offset,
                        'process_time': op['process_time']
                    }
                    self.originActionMask[action_id] = 1
                self.reward_denominator += op["MPT"]
        
        if self.noop is True:
            self.action[-1] = {
                "job_type": "noop",
                "op_id": "noop"
            }
        # for id, action in enumerate(self.action):
        #     if self.originActionMask[id] == 1:
        #         print('action {}, O({}, {}) on M{}'.format(id, action['job_type'], action['op_id'], action['machine_id']))
        # print(self.originActionMask)
        self.reward_denominator /= (self.action_space.n - 1)
        self.action_mask = np.zeros(self.action_space.n)

    def save_instance(self, file_path):
        self.DJSP_Instance.save_instance(file_path)

    def load_instance(self, file_path, noop):
        self.DJSP_Instance.load_instance(file_path)
        self.noop = noop
        self.__define_action()
        self.__preprocess_state()
        self.noop = noop
        self.restart()

    def load_basic_type(self, dir = None):
        # file_name = os.path.join(case_dir, 'basic.json')
        file_name = self.job_type_file if dir is None else os.path.join(dir, 'basic.json')
        self.load_instance(file_name, self.noop)
    
    def load_evaluation(self, file_path):
        self.DJSP_Instance.load_instance(file_path)
        self.evaluation_file = file_path
        self.evaluation_flag = True
    
    def train_mode(self):
        self.evaluation_file = ""
        self.evaluation_flag = False

    def __action_id(self, job_offset, op_id, machine_id):
        return job_offset * self.maxOPperJob * self.maxMachineNum +\
                op_id * self.maxMachineNum + machine_id
    
    def P__action_id(self, job_offset, op_id, machine_id):
        return self.__action_id(job_offset, op_id, machine_id)

    def __maskAction2machine(self, action_id):
        return action_id % self.maxMachineNum

    def close(self):
        pass

if __name__ == '__main__':
    env_config = {
        "djspArgsFile": './args.json',
        "noop": False,
        "fixed_case": True
    }
    env = DJSP_Env2(env_config)
    env.DJSP_Instance.show()
    env.DJSP_Instance.show_registedJobs()
    # # env.load_evaluation('./test_instance/Case4/validate_4.json')
    # # env.DJSP_Instance.show()
    input()
    obs = env.reset()
    done = False
    while not done:
        candidate = []
        for id in range(env.action_space.n):
            if env.action_mask[id] == 1:
                candidate.append(id)
        action = choice(candidate)
        _obs, reward, done, info = env.step(action)
        print('available action len: ', len(candidate))
    print('done')
    print('check: ', env.DJSP_Instance.check_schedule())
    print('tardiness: ', env.DJSP_Instance.Tardiness())