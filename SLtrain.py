import numpy as np
from model import PolicyNetwork
from dataclasses import dataclass
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from google.colab import drive
# drive.mount("/content/drive")


epoch_size = 30  # 2 # 130
minibatch_size = 4096  # 100  # 4096


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
    # state[0][0], state[0][1] = state[0][1], state[0][0]
    state = np.array([state[0][[1, 0], :]])
    return state


def to_train_data(t_data_pre):  # train_data_pre -> train_data
    t_data = train_data(to_state(t_data_pre.state_pre), t_data_pre.action)
    if t_data_pre.color == "white":
        t_data.state = to_state_opposite(t_data.state)
    return t_data


def load_sl_train_data_pre():
    print("[INFO] Start loading train data...")
    # with open("train_data_for_test.txt", "r") as f:  # for test
    # with open("/content/drive/My Drive/train_data.txt", "r") as f:  # for colab
    with open("train_data.txt", "r") as f:
        data = f.readlines()
    ans = []
    for _, line in enumerate(tqdm(data)):
        if "B" in line:
            ans.append(to_train_data_pre(line, "black"))
        elif "W" in line:
            ans.append(to_train_data_pre(line, "white"))
    return ans


def convert(data):
    state = np.array([np.fliplr(np.flipud(data.state[0][0])), np.fliplr(np.flipud(data.state[0][1]))])
    state = state.reshape((1, 2, 8, 8))
    action = (7 - data.action[0], 7 - data.action[1])

    return train_data(state, action)


def rotate_train_data(train_data_):
    action = train_data_.action
    state = train_data_.state
    state = np.rot90(state, k=1, axes=(2, 3)).copy()
    action = action[1], int(-1 * (action[0] - 3.5) + 3.5)
    return train_data(state, action)


def train_PolicyNetwork(network):
    # writer = SummaryWriter(log_dir="/content/drive/My Drive/logs")  # tensorboard for colab
    writer = SummaryWriter(log_dir="./logs")  # tensorboard
    writer.add_graph(network, torch.from_numpy(np.zeros(shape=(1, 2, 8, 8))).double().to(dev))

    t_data_pre = load_sl_train_data_pre()  # 学習用データの準備
    t_data_pre_for_test = np.random.choice(t_data_pre, 500, replace=False)  # 500, 50 : for test
    t_data_for_test = [convert(to_train_data(data)) for data in t_data_pre_for_test]

    loop_size = 30  # 1 epoch 内での学習回数
    t_data_pre_size = len(t_data_pre)
    optimizer = torch.optim.Adam(network.parameters())
    lossfunc = torch.nn.CrossEntropyLoss()
    train_datas = [to_train_data(data) for data in tqdm(t_data_pre)]
    del t_data_pre  # メモリ解放

    # test = train_datas[0]
    # print("test : {}".format(test))
    # test2 = rotate_train_data(test)
    # print("test2 : {}".format(test2))
    print("[INFO] Start train.")

    for i, epoch in enumerate(tqdm(range(epoch_size))):
        # random_index = np.random.choice(t_data_pre_size, loop_size, replace=False)
        losses = []
        # for _, idx in enumerate(tqdm(random_index)):
        for _ in enumerate(tqdm(range(loop_size))):
            # datas = t_data_pre[idx:min(idx + minibatch_size, t_data_pre_size)]
            datas = np.random.choice(train_datas, minibatch_size)
            # datas = [to_train_data(data) for data in datas]
            loss = torch.tensor(0.0, requires_grad=True).to(dev)
            for data in datas:
                predict = network(torch.from_numpy(data.state).double().to(dev))
                action = torch.Tensor([data.action[0] * 8 + data.action[1]]).long().to(dev)
                if action < 0:
                    print("[ERROR] action id ({}, {}), {}.".format(data.action[0], data.action[1], action))  # debug
                else:
                    loss = (loss + lossfunc(predict, action)).to(dev)
            optimizer.zero_grad()
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

        tmp = 0
        network.eval()
        for data in t_data_for_test:  # test
            pred = network(torch.from_numpy(data.state).double().to(dev)).cpu().detach().numpy().argmax()
            tmp += int((pred // 8, pred % 8) == data.action)
        network.train()

        accuracy = float(tmp) / float(len(t_data_for_test)) * 100.0
        ave_loss = sum(losses) / len(losses) / minibatch_size
        print("[INFO] (epoch {}) accuracy rate is {}%, loss ave is {}.".format(epoch, accuracy, ave_loss))
        writer.add_scalar("Average loss", ave_loss, i)
        writer.add_scalar("Accuracy", accuracy, i)
        network = network.cpu()
        # torch.save(network.state_dict(), "/content/drive/My Drive/SLNet.pth")
        torch.save(network.state_dict(), "./models/SLNet.pth")
        network = network.to(dev)
    writer.close()
    print("[INFO] End SL training.")


def main():
    print("test")
    net = PolicyNetwork().double().to(dev)
    train_PolicyNetwork(net)


if __name__ == "__main__":
    main()
