import torch
import torch.nn as nn
from board import othello

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        kernel_size = 3
        out_channels = 128
        self.c1 = nn.Conv2d(2, out_channels, kernel_size, padding=1)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.c3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.c4 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.c5 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.c6 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.c7 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.c8 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        # self.c9 = nn.Conv2d(out_channels, 1, kernel_size, padding=1)
        self.c9 = nn.Conv2d(out_channels, 1, 1)
        out_shape = 64 * 8 * 8
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=0.1)
        self.train()

    def forward(self, state):
        x = self.relu(self.c1(state))
        # print(x.shape)
        x = self.relu(self.c2(x))
        # print(x.shape)
        x = self.relu(self.c3(x))
        x = self.relu(self.c4(x))
        x = self.relu(self.c5(x))
        x = self.relu(self.c6(x))
        x = self.relu(self.c7(x))
        x = self.relu(self.c8(x))
        x_1 = self.c9(x)
        # x_1 = x_1.view(-1, 64 * 8 * 8)
        # x_1 = self.l1(x_1)
        x_1 = x_1.view(-1, 64)
        x_1 = self.drop(self.l1(x_1))
        x_1 = self.l2(x_1)
        # x_1 = self.softmax(x_1)
        return x_1


def main():
    print("test")
    bd = othello()
    # slnet = SLNetwork().double()
    inp = torch.from_numpy(bd.get_state()).double()
    # print(inp.dtype)
    # x = slnet(inp)
    # print(x)
    anet = ValueNetwork().double()
    x = anet(inp)
    print(x)


if __name__ == "__main__":
    main()

