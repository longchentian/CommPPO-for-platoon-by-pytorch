import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        # self.s = np.zeros((args.batch_size+1, args.state_dim))
        # self.a = np.zeros((args.batch_size+1, args.action_dim))
        # self.a_logprob = np.zeros((args.batch_size+1, args.action_dim))
        # self.r = np.zeros((args.batch_size+1, 1))
        # self.s_ = np.zeros((args.batch_size+1, args.state_dim))
        # self.dw = np.zeros((args.batch_size+1, 1))
        # self.done = np.zeros((args.batch_size+1, 1))
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0
        self.use_cuda = args.use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device('cuda:0')
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            self.device = torch.device('cpu')
            print("Device set to : cpu")

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        # s = torch.tensor(self.s, dtype=torch.float)
        # a = torch.tensor(self.a, dtype=torch.float)
        # a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        # r = torch.tensor(self.r, dtype=torch.float)
        # s_ = torch.tensor(self.s_, dtype=torch.float)
        # dw = torch.tensor(self.dw, dtype=torch.float)
        # done = torch.tensor(self.done, dtype=torch.float)

        s = torch.tensor(self.s, dtype=torch.float).to(self.device)
        a = torch.tensor(self.a, dtype=torch.float).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)

        return s, a, a_logprob, r, s_, dw, done
