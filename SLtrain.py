import numpy as np
from model import PolicyNetwork
from dataclasses import dataclass
from tqdm import tqdm
import torch

epoch_size = 120  # 30
minibatch_size = 1096  # 4096


@dataclass
class train_data_pre:
    state_pre: np.array  # (8, 8)
    action: tuple  # (int, int)
    color: str  # black / white


@dataclass
class train_data:
    state: np.array  # (1, 2, 8, 8)
    action: tuple  # (int, int)


def to_train_data_pre(string, color):  # train_data_pre に変換する
    tmp = string.replace("\n", "").split(" ")
    state_pre = np.array([float(tmp[i]) for i in range(64)]).reshape(8, 8)
    act = (int(tmp[64]) - 1, int(tmp[65]) - 1)
    ans = train_data_pre(state_pre, act, color)
    return ans


def to_state(state_pre):  # state_pre -> state の変換
    ans = np.zeros((1, 2, 8, 8))
    for i in range(8):
        for j in range(8):
            if state_pre[i][j] == 1:
                ans[0][1][i][j] = 1.0
            elif state_pre[i][j] == 2:
                ans[0][0][i][j] = 1.0
    return ans


def to_state_opposite(state):  # white視点のボードをblack視点に変換する
    state[0][0], state[0][1] = state[0][1], state[0][0]
    return state


def to_train_data(t_data_pre):  # train_data_pre -> train_data
    t_data = train_data(to_state(t_data_pre.state_pre), t_data_pre.action)
    if t_data_pre.color == "white":
        t_data.state = to_state_opposite(t_data.state)
    return t_data


def load_sl_train_data_pre():
    print("[INFO] Start loading train data...")
    # with open("train_data_for_test.txt", "r") as f:
    with open("train_data.txt", "r") as f:
        data = f.readlines()
    ans = []
    for _, line in enumerate(tqdm(data)):
        if "B" in line:
            ans.append(to_train_data_pre(line, "black"))
        elif "W" in line:
            ans.append(to_train_data_pre(line, "white"))
    return ans


def train_PolicyNetwork(network):
    t_data_pre = load_sl_train_data_pre()  # 学習用データの準備
    data_size = len(t_data_pre) // 200000
    t_data_pre_size = len(t_data_pre)
    optimizer = torch.optim.Adam(network.parameters(), weight_decay=5e-4)
    lossfunc = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epoch_size)):
        random_index = np.random.choice(t_data_pre_size, data_size, replace=False)
        # datas = np.random.choice(t_data_pre, minibatch_size, replace=False)  # randomにminibatch_size個を重複なしで取り出す.
        # datas = [data for data in random_index]
        # datas = [to_train_data(data) for data in datas]
        accuracy_counter = 0
        datas_len = 0
        for _, idx in enumerate(tqdm(random_index)):
            datas = t_data_pre[idx:min(idx + minibatch_size, t_data_pre_size)]
            datas = [to_train_data(data) for data in datas]
            datas_len += len(datas)
            loss = torch.tensor(0.0, requires_grad=True)
            for data in datas:
                predict = network(torch.from_numpy(data.state).double())
                action = torch.Tensor([data.action[0]*8 + data.action[1]]).long()
                if action < 0:
                    print("action id ({}, {}), {}.".format(data.action[0], data.action[1], action))
                else:
                    loss = loss + lossfunc(predict, action)
                    pred = predict.detach().numpy()
                    pred = (pred.argmax() // 8, pred.argmax() % 8)
                    if pred == data.action:
                        accuracy_counter += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[INFO] accuracy rate is {}%.".format(float(accuracy_counter) / float(datas_len) * 100.0))

        print("[INFO] accuracy rate is {}% in epoch {}".format(float(accuracy_counter) / float(datas_len) * 100.0, epoch))
        print("[INFO] end train SL Network.")
        torch.save(network.state_dict(), "./SLNet.pth")
    print("[INFO] SL Network model is saved.")


def main():
    print("test")
    net = PolicyNetwork().double()
    train_PolicyNetwork(net)


if __name__ == "__main__":
    main()
