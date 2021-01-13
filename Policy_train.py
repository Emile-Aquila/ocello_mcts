from my_REINFORCE import REINFORCE
from model import PolicyNetwork
import torch
from env import Environment, select_legal_hand
from tqdm import tqdm
import numpy as np

train_times = 1000
train_epoch_size = 32  # 相互対戦する回数
max_len = 5


def transform_state(state, your_color):  # color="black"ならそのままstateを返す. "white"ならstateを変換して返す
    if your_color == "white":
        return np.array([state[1], state[0]])
    else:
        return state


def policy_train():
    opposite_networks = [PolicyNetwork().double()]
    train_net = PolicyNetwork().double()
    agent = REINFORCE(train_net, torch.optim.Adam())  # 学習するagent
    env = Environment(opposite_networks[0], "black")
    total_play_times = 0
    total_win_times = 0

    for j in tqdm(range(train_times)):
        op_net = np.random.choice(opposite_networks, 1)  # opposite_networksから一つ,対戦相手のnetworkを選択.
        for i in range(train_epoch_size):
            your_color = "black" if (i % 2 == 0) else "white"  # 学習するnetの色
            env.reset(op_net, your_color=your_color)  # envの初期化
            while True:
                state = torch.from_numpy(transform_state(env.get_state(), your_color)).double()
                probs = agent.act(state)
                action = select_legal_hand(probs, env.othello.legal_hands(your_color))
                n_state, rew, done = env.step(action, show=(i+1 == train_epoch_size))
                n_state = torch.from_numpy(transform_state(n_state, your_color)).double()
                agent.observe(n_state, rew, done, False)
                if done:
                    if rew == 1.0:
                        total_win_times += 1
                    total_play_times += 1
                    break
            agent.batch_update()
        if len(opposite_networks) > max_len:
            opposite_networks.pop()
        opposite_networks.append(agent.get_model().copy())
        print("[INFO] win rate is {}, loop{} end".format(float(total_win_times) / float(total_play_times), j))
    return agent.model()


def main():
    print("test")
    policy_train()


if __name__ == "__main__":
    main()
