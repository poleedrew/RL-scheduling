from ENV.DJSP_Env2 import DJSP_Env2
from ENV.utils.Job_type_DJSP import Job_type_DJSP
from Args import args
import os

if __name__ == '__main__':
    DJSP_config = {
        'MachineNum': 5,
        'OpMachineNum': 5,
        'InitialJobNum': 5,
        'InsertedJobNum': 0,
        'DDT': 1.5,
        'DDT_dt': 0.8,
        'MaxOpNum': 3,
        'MaxProcessTime': 50,
        'RPT_effect': False,
        'Mean_of_Arrival': 25,
        'JobTypeNum': 5,
        'JobTypeFile': ""
    }
    Instance = Job_type_DJSP(
        DJSP_config
    )
    # env = DJSP_Env2(
    #     DJSP_config = DJSP_config,
    #     noop = False
    # )
    if os.path.exists(args.djsp_instance) is True:
        print('file/dir %s exists.' % args.djsp_instance)
        print('set test_case directory in arg --DJSP_INSTANCE')
    else:
        os.mkdir(args.djsp_instance)
        Instance.save_instance(os.path.join(args.djsp_instance, 'basic.json'))
        # env.DJSP_Instance.Fixed_type = False # random pick job type
        Instance.show()
        input()
        Instance.updateJobNum(10, 40)
        for i in range(args.instance_num):
            
            Instance.restart(refresh=True)
            Instance.save_instance(os.path.join(args.djsp_instance, 'validate_%d.json' % (i + 1)))
            Instance.show_registedJobs()
