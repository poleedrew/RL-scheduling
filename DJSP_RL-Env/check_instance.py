from Args import args
from ENV.utils.Job_type_DJSP import Job_type_DJSP
import os

if __name__ == '__main__':
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
    Test_instance.show()
    Test_instance.show_registedJobs()