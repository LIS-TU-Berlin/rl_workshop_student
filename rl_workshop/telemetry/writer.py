import yaml
from torch.utils.tensorboard import SummaryWriter
from rl_workshop.paths import tensorboard_dir

class Writer:
    def __init__(self, tag):
        # self.ep_length = [0., 0.]
        # self.ep_sum_reward = [0., 0.]
        # self.ep_dis_return = [0., 0.]
        # self.eval_length = [0., 0.]
        # self.eval_sum_reward = [0., 0.]
        # self.eval_dis_return = [0., 0.]
        # self.pi_value = [0., 0.]
        # self.pi_BC_loss = [0., 0.]
        # self.Q_loss = [0., 0.]
        self.data = {}
        self.tag = tag
        self.writer = SummaryWriter(log_dir=f"{tensorboard_dir()}/{self.tag}")

    def add(self, key, value):
        if not key in self.data:
            self.data[key] = [0.0, 0.0]
        self.data[key][0] += float(value)
        self.data[key][1] += 1.0

    def get(self):
        dat = {}
        for key, value in self.data.items():
            if value[1] > 0.0:
                dat[key] = value[0] / value[1]
                self.data[key] = [0.0, 0.0]
        return dat

    def write(self, t, verbose=0):
        D = self.get()
        for key, value in D.items():
            self.writer.add_scalar(f"{key}", value, t)
        self.writer.flush()

        if verbose > 0:
            print(f"   Writer {self.tag} t:{t} data:\n{yaml.dump(D)}", end='')
