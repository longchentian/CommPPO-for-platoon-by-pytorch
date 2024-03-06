import torch
import numpy as np
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous

import optparse
import traci
import matplotlib.pyplot as plt
import datetime

Platoonsize_Max = 16

episode_num = 0
episode_max = 300
# episode_max = 10
step_max = 140
BATCH = 16

import sys
import os
import math

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable ")

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def state_calc(vehicle, position, speed, acc):
    '''
        自己参考其他论文编写的状态空间，太痛苦了没法收敛
    '''
    state = [0] * 6
    # state[0] = position[vehicle + 1]/100
    # state[1] = speed[vehicle + 1]/20
    # state[2] = position[vehicle]/100
    # state[3] = speed[vehicle]/20
    # state[4] = position[0]/100
    # state[5] = speed[0]/20
    state[0] = position[vehicle + 1]
    state[1] = speed[vehicle + 1]
    state[2] = position[vehicle]
    state[3] = speed[vehicle]
    state[4] = position[0]
    state[5] = speed[0]
    return state

def reward_calc(vehicle, position, speed, acc, acc_conflict, pos_conflict,acc_last = [], Platoonsize_Max = 16):

    t1 = 0.35
    t2 = 0.25
    t3 = 0.45
    t4 = 0.35
    t5 = 0.15
    r1 = abs(position[vehicle] - position[vehicle+1] - 10 - (speed[vehicle+1] - speed[vehicle]))*t1
    r2 = abs(position[0] - position[vehicle + 1] - 10 * (vehicle + 1) - (vehicle + 1) *(speed[vehicle+1] - speed[0]))*t2
    r3 = abs(speed[vehicle] - speed[vehicle+1])*t3
    r4 = abs(speed[0] - speed[vehicle+1])*t4
    r5 = abs(acc[vehicle] - acc_last[vehicle+1])*t5
    r6 = 0
    if acc_conflict == 1 or pos_conflict == 1:
        # 发生冲突或者超车,奖励值 + 1
        r6 = 5
    reward = -(r1 + r2 + r3 + r4 + r5 + r6)
    reward *= 20

    return reward




def main(args, seed):
    m = 1200
    Af = 2.5
    Cd = 0.32
    rouair = 1.184
    g = 9.8
    speedmode = 6
    miu = 0.015
    con = []
    ave = 20
    madr = 1.4
    sumoBinary = checkBinary('sumo')
    # sumoBinary = checkBinary('sumo-gui')
    speed_init = 20
    leading = []
    for i in range(0, 40):
        leading.append(0)
    for i in range(40, 70):
        leading.append(-1)
    for i in range(70, 150):
        leading.append(1)
    for i in range(150, 300):
        leading.append(0)

    episode_reward = [0]
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = 6
    args.action_dim = 1
    args.max_action = 3
    args.max_episode_steps = 300
    # args.max_episode_steps = 10

    args.batch_size = 1 * 16
    args.mini_batch_size = 1
    replay_buffer = ReplayBuffer(args)
    # agent = PPO_continuous(args)
    ppo = PPO_continuous(args)
    path1 = './models/ACTOR_other.pth'
    path2 = './models/CRITIC_other.pth'

    ppo.load_model(path1,path2)

    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # while total_steps < args.max_train_steps:



    for episode in range(episode_max * 5):
        if episode // episode_max >= 0:
            Platoonsize = 1
        if episode // episode_max >= 1:
            Platoonsize = 2
        if episode // episode_max >= 2:
            Platoonsize = 4
        if episode // episode_max >= 3:
            Platoonsize = 8
        if episode // episode_max >= 4:
            Platoonsize = 16

        if episode == episode_max:
            # replay_buffer.count = 0
            args.batch_size = Platoonsize * 16
            args.mini_batch_size = Platoonsize
            replay_buffer = ReplayBuffer(args)
            ppo.batch_size = Platoonsize * 16
            ppo.mini_batch_size = Platoonsize

        elif episode == 2 * episode:
            args.batch_size = Platoonsize * 16
            args.mini_batch_size = Platoonsize
            replay_buffer = ReplayBuffer(args)
            ppo.batch_size = Platoonsize * 16
            ppo.mini_batch_size = Platoonsize

        elif episode == 3 * episode:
            args.batch_size = Platoonsize * 16
            args.mini_batch_size = Platoonsize
            replay_buffer = ReplayBuffer(args)
            ppo.batch_size = Platoonsize * 16
            ppo.mini_batch_size = Platoonsize

        elif episode == 4 * episode:
            args.batch_size = Platoonsize * 16
            args.mini_batch_size = Platoonsize
            replay_buffer = ReplayBuffer(args)
            ppo.batch_size = Platoonsize * 16
            ppo.mini_batch_size = Platoonsize

        Position = [0] * (Platoonsize + 1)
        Speed = [0] * (Platoonsize + 1)
        Acc = [0] * (Platoonsize + 1)
        position_plot = []
        speed_plot = []
        acc_plot = []
        time_plot = []

        reward_ave = 0
        done = 0
        consumption = 0
        distance = 0
        # traci.start([sumoBinary, "-c", "car_str.sumocfg"])
        traci.start([sumoBinary, "-c", "car_str_16.sumocfg"])
        state = [[] for i in range(Platoonsize + 1)]
        reward = [[] for i in range(Platoonsize + 1)]
        action = [[] for i in range(Platoonsize + 1)]
        action_logprob = [[] for i in range(Platoonsize + 1)]
        state_next = [[] for i in range(Platoonsize + 1)]
        buffer_reward = [[] for i in range(Platoonsize + 1)]
        # 初始化先跑一秒，才能读到车辆
        traci.simulationStep()
        exist_list = traci.vehicle.getIDList()
        for car in exist_list:
            ind = exist_list.index(car)
            if ind <= Platoonsize:
                # if car != 'a':
                #     traci.vehicle.setSpeedMode(car, speedmode)
                Position[ind] = traci.vehicle.getPosition(car)[0]
                Speed[ind] = traci.vehicle.getSpeed(car)
                Acc[ind] = traci.vehicle.getAcceleration(car)
        accelerate_accepted = [3] * (Platoonsize + 1)

        for step in range(step_max):

            for i in range(Platoonsize + 1):
                position_plot.append(Position[i] / 1000)
                speed_plot.append(Speed[i])
                time_plot.append(step)
            # 得到限定加速度区间
            for i in range(Platoonsize):
                if Speed[i] - 3 < Speed[i + 1]:
                    # 前车速度小于后车速度3，就是说后车加速度3的话可能碰撞 gap<0就要减速
                    gap = Position[i] - Position[i + 1] - 5 - Speed[i + 1] + max(Speed[i] - 3, 0)
                    if gap < 0:
                        amax = -3
                    else:
                        amax = min(gap / 3, math.sqrt(madr * gap)) + Speed[i] - Speed[i + 1] - 3
                        amax = np.clip(amax, -3, 3)
                else:
                    amax = 3

                accelerate_accepted[i + 1] = amax

            acc_matrix = [0] * (Platoonsize + 1)

            # 状态值获取

            for i in range(1, Platoonsize + 1):
                state[i].append(state_calc(i - 1, Position, Speed, Acc))

            for i in range(1, Platoonsize + 1):
                # 这里一定要是-1,因为状态是存储了多个时间步的
                act, act_log_prob = ppo.choose_action(np.array(state[i][-1]))
                if args.policy_dist == "Beta":
                    act = 3 * (act - 0.5) * args.max_action  # [0,1]->[-max,max][-3,3]

                action[i].append(act)
                action_logprob[i].append(act_log_prob)

            for i in range(1, Platoonsize + 1):
                acc_matrix[i] = action[i][-1][0]

            acc_conflict = [0] * (Platoonsize + 1)
            pos_conflict = [0] * (Platoonsize + 1)
            # 作者在原文这里添加了领航车的速度处理
            speed_next = np.clip(Speed[0] + leading[step], 0, speed_init)
            # traci.vehicle.setSpeed(exist_list[0], speed_next)

            for i in range(1, Platoonsize + 1):
                print(exist_list)
                # 他是按照倒立摆的DPPO模板改的，里面最大输出动作是2，要输出最大加速度在-3到3之间，所以要乘以1.5
                accc = min(acc_matrix[i], accelerate_accepted[i])

                # if acc_matrix[i] > accelerate_accepted[i] + 0.5:
                if acc_matrix[i] > accelerate_accepted[i]:
                    # 这里应该就是a_conflict
                    acc_conflict[i] = 1
                speed_next = np.clip(Speed[i] + accc, 0, 33)
                if i < Platoonsize + 1:
                    # 前期碰撞先设置固定加速度，防止碰撞后无法进行下一步
                    traci.vehicle.setSpeed(exist_list[i], speed_next)

            Acc_last = Acc
            Position = [0] * (Platoonsize + 1)
            Speed = [0] * (Platoonsize + 1)
            Acc = [0] * (Platoonsize + 1)

            traci.simulationStep()

            exist_list = traci.vehicle.getIDList()

            if 'a' not in exist_list:
                # print('a not find')
                # traci.close()
                #这里要提前结束了
                if episode % episode_max == 0:
                    episode_reward[0] = reward_ave / step / Platoonsize
                else:
                    reward_ave = reward_ave / step / Platoonsize * 0.8 + episode_reward[-1]*0.2 # 让曲线更平滑
                    episode_reward.append(reward_ave)
                break

            for car in exist_list:
                ind = exist_list.index(car)
                # print(car)
                if ind <= Platoonsize:
                    Position[ind] = traci.vehicle.getPosition(car)[0]
                    Speed[ind] = traci.vehicle.getSpeed(car)
                    Acc[ind] = traci.vehicle.getAcceleration(car)

            for i in range(1, Platoonsize + 1):
                # 这里判断是否发生碰撞，碰撞的话车会消失，所以一般要结束进程了，或许前期可以通过算法挽回以下，让他跑更多的时间步
                if Position[i] >= Position[i - 1] - 5 or Position[i] <= -10000:
                    pos_conflict[i] = 1

            # 下一时刻状态值获取
            for i in range(1, Platoonsize + 1):
                state_next[i].append(state_calc(i - 1, Position, Speed, Acc))

            for i in range(1, Platoonsize + 1):
                reward[i].append(reward_calc(i - 1, Position,
                                             Speed, Acc,
                                             acc_conflict[i], pos_conflict[i],
                                             Acc_last, Platoonsize))
                # episode_reward[i] += reward[i][-1]

            for i in range(1, Platoonsize + 1):
                # buffer_reward[i].append((reward[i][-1] + ave) / ave)
                buffer_reward[i].append(reward[i][-1])
                reward_ave += reward[i][-1]

            for i in range(1, Platoonsize + 1):
                replay_buffer.store(state[i][-1],action[i][-1],action_logprob[i][-1],buffer_reward[i][-1],state_next[i][-1],False,0)

            if replay_buffer.count >= args.batch_size:
                ppo.update(replay_buffer, episode-episode//episode_max * episode_max)
                replay_buffer.count = 0
            if sum(pos_conflict) > 0:
                # traci.close()
                sys.stdout.flush()
                print('车辆发生碰撞')
                print(step, pos_conflict)
                if episode % episode_max == 0:
                    episode_reward[0] = reward_ave / step / Platoonsize
                else:
                    reward_ave = reward_ave / step / Platoonsize * 0.8 + episode_reward[-1]*0.2 # 让曲线更平滑
                    episode_reward.append(reward_ave)
                break


            if step >= step_max - 1:
                # 当达到最大步数前 领航车可能会消失，所以 直接关sumo
                # traci.close()
                # reward_ave = reward_ave / step_max / Platoonsize
                # episode.append(reward_ave
                if episode % episode_max == 0:
                    episode_reward[0] = reward_ave / step_max / Platoonsize
                else:
                    reward_ave = reward_ave / step_max / Platoonsize * 0.8 + episode_reward[-1]*0.2 # 让曲线更平滑
                    episode_reward.append(reward_ave)
                break

        traci.close()
        if episode % 10 == 0:
            # plt.ion()
            plt.figure(0)
            plt.scatter(time_plot, position_plot, c=speed_plot, s=10, alpha=0.3)
            plt.colorbar()
            plt.xlabel('Time (s)')
            plt.ylabel('Location (km)')
            plt.grid(True)
            # plt.show()
            mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')  # 这里是引用时间
            plt.savefig('.\\output\\Location time{}.png'.format(mkfile_time), dpi=300)  # 分别创建文件夹，分别储存命名图片
            plt.close('all')
        if (episode+1) % episode_max ==0:
            # plt.ion()
            plt.figure(1)
            plt.plot(np.arange(len(episode_reward)),episode_reward)
            plt.xlabel('episode')
            plt.ylabel('reward')
            plt.grid(True)
            plt.show()
            mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')  # 这里是引用时间
            plt.savefig('.\\output\\Reward episode{}.png'.format(mkfile_time), dpi=300)  # 分别创建文件夹，分别储存命名图片
            episode_reward = [0]
            replay_buffer.count = 0
            plt.close('all')
        if episode % 10 == 0:
            ppo.save_model()



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(300), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=100, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=0.0001, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=0.0002, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_cuda", type=bool, default=False, help="cuda")
    args = parser.parse_args()


    main(args, seed=10)
