import numpy as np 
import sys
# from Args import args

def GenOperation(op_num, machine_num, machine_num_dist_op, process_time_dist):
    op = []
    for op_id in range(op_num):
        candidate_num = np.random.randint(*machine_num_dist_op)
        # expected_process_time = np.random.uniform(*process_time_dist) # continuous
        m_set = np.sort(np.random.choice(machine_num, size=candidate_num, replace=False))
        expected_process_time = [-1 for _ in range(machine_num)]
        MPT = 0
        for m in m_set:
            expected_process_time[m] = np.random.uniform(*process_time_dist)
            MPT += expected_process_time[m]
        op.append({
            "id": op_id,
            "candidate_machine": m_set,
            "process_time": np.array(expected_process_time),
            "MPT": MPT/len(m_set)
        })
    return op
    
def GenByRandom(job_num, machine_num, process_time_range, op_num):
    jobs = []
    for i in range(job_num):
        single_job = []
        for _ in range(op_num):
            single_job.append([[np.random.randint(machine_num)], np.random.randint(*process_time_range)])
        jobs.append(single_job)

    return job_num, machine_num, jobs

def GenByConst():
    job_num, op_num, machine_num = 4, 3, 3
    jobs = [
        [[[0], 5], [[0], 8], [[0], 9]],
        [[[1], 6], [[0], 3], [[2], 4]],
        [[[2], 7], [[1], 5], [[2], 5]],
        [[[1], 15], [[1], 8], [[0], 9]]
    ]
    return job_num, machine_num, jobs

if __name__ == '__main__':
    print(args.file)
    if args.file is not None:
        job_num, machine_num, jobs = GenByFile(args.file)