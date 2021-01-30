from my_REINFORCE import REINFORCE
from model import PolicyNetwork
import torch
from env import Environment, select_legal_hand
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from board import check_set_stone

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_times = 1000
train_epoch_size = 32  # 相互対戦する回数
max_len = 50


def transform_state(state, your_color):  # color="black"ならそのままstateを返す. "white"ならstateを変換して返す
    if your_color == "white":
        tmp = np.array([[state[0][1], state[0][0]]])
        return tmp
    else:
        return state


def policy_train():
    opposite_networks = [PolicyNetwork().double()]
    train_net = PolicyNetwork().double().to(dev)
    train_net.load_state_dict(torch.load("./SLnet_trained.pth"))
    agent = REINFORCE(train_net, torch.optim.Adam(train_net.parameters()))  # 学習するagent
    env = Environment(opposite_networks[0], "black")

    for j in tqdm(range(train_times)):
        op_net = np.random.choice(opposite_networks, 1)  # opposite_networksから一つ,対戦相手のnetworkを選択.
        total_play_times, total_win_times = 0, 0
        for i in range(train_epoch_size):
            your_color = "black" if (i % 2 == 0) else "white"  # 学習するnetの色
            env.reset(op_net, your_color=your_color)  # envの初期化
            while True:
                state = torch.from_numpy(transform_state(env.get_state(), your_color)).double()
                action = agent.act(state[0])
                # if i+1 == train_epoch_size:
                #     print("action {},{}. ".format(action // 8, action % 8))
                n_state, rew, done = env.step(action)
                # n_state, rew, done = env.step(action, show=(i+1 == train_epoch_size))
                n_state = torch.from_numpy(transform_state(n_state, your_color)).double()
                # print("rew,done is {},{}".format(rew, done))
                agent.observe(n_state, rew, done, done)
                if done:
                    if rew == 1.0:
                        total_win_times += 1
                    total_play_times += 1
                    break
            # agent.batch_update()
        if len(opposite_networks) > max_len:
            opposite_networks.pop()
        win_rate = total_win_times / total_play_times
        print("[INFO] win rate is {}, loop:{} end".format(win_rate, j))
        opposite_networks.append(deepcopy(agent.get_model()))
        if win_rate > 0.5:
            torch.save(agent.get_model().state_dict(), "./models/PolicyNet.pth")
    return agent.model()


def main():
    print("test")
    policy_train()


if __name__ == "__main__":
    main()
