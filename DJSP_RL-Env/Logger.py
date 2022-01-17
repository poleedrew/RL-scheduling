from torch.utils.tensorboard import SummaryWriter
import os, time
from Args import args

class Logger:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.dir_path, self.log_path ,self.model_path = self.init_dir()
        self.record_experiment_setting()
        self.writer = SummaryWriter(self.log_path)
    
    def record(self, key, value, epoch):
        self.writer.add_scalar(key, value, epoch)

    def record_experiment_setting(self):
        fp = open(os.path.join(self.dir_path, 'setting.txt'), "w")
        for arg in vars(args):
            fp.write("arg: {:20}, value: {}\n".format(arg, getattr(args, arg)))
        fp.close()

    def init_dir(self):
        timestamp = time.time()#time.localtime()
        dir_name = "{}".format(timestamp) if self.prefix == "" else "{}_{}".format(self.prefix, timestamp)
        path = os.path.join(os.path.expanduser("~"), 'experiment', dir_name)
        log_path = os.path.join(path, "log")
        model_path = os.path.join(path, "model")
        os.makedirs(path)
        os.makedirs(log_path)
        os.makedirs(model_path)
    
        return path, log_path, model_path

if __name__ == '__main__':
    test = Logger()