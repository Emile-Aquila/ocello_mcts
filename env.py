import torch
from board import othello
import numpy as np
import torch
import torch.nn.functional as F

#
# def change_coordinate(x, y):  # (x,y) -> 8*x + y に変換
#     return x * 8 + y


def change_coordinate(coordinate):  # 8*x + y -> (x,y) に変換
    return coordinate // 8, coordinate % 8


def select_legal_hand(probs, legal_hands):  # 合法手の中で確率的に手を選択する
    probs_in_legal = []  # legal_hands に含まれる probs の要素のリスト
    index_in_legal = []  # legal_hands に含まれる probs の要素の,元の配列に置ける index のリスト
    for legal_hand in legal_hands:
        legal_coord = change_coordinate(legal_hand)
        # print("legal hand {}, probs {}".format(legal_hand, probs))
        probs_in_legal.append(probs[0][legal_hand].detach().item())
        index_in_legal.append(legal_hand)
    # print("prob in legal {}".format(probs_in_legal))
    sum_of_probs = sum(probs_in_legal)
    probs_in_legal = [i / sum_of_probs for i in probs_in_legal]
    ans = np.random.choice(len(probs_in_legal), size=1, p=probs_in_legal)[0]
    # print("ans {}".format(ans))
    return index_in_legal[ans]  # coordinate の形で座標を返す


class Environment:
    def __init__(self, policy_net, your_color="black"):
        self.othello = othello()  # オセロ盤
        self.opposite_net = policy_net  # 対戦相手側のAI
        self.your_color = your_color  # プレイヤー側の色
        self.opposite_color = "white" if self.your_color == "black" else "black"  # 対戦相手の色
        self._state = self.othello.get_state()
        self.softmax = torch.nn.Softmax(dim=1)
        if self.your_color == "white":
            state_tmp = torch.from_numpy(np.array([self._state[1], self._state[0]])).double()  # 白黒反転させたstateの算出
            probs = self.softmax(self.opposite_net(state_tmp))
            action = select_legal_hand(probs, self.othello.legal_hands(self.opposite_color))  # 行動選択
            self.othello.set_stone(action, self.opposite_color)  # 石を置く.
            self._state = self.othello.get_state()  # 状態の更新

    def reset(self, agent_net, your_color="black"):
        self.othello.reset()
        self.opposite_net = agent_net[0]
        self.your_color = your_color
        self.opposite_color = "white" if self.your_color == "black" else "black"  # 対戦相手の色
        self._state = self.othello.get_state()
        if self.your_color == "white":
            state_tmp = torch.from_numpy(np.array([[self._state[0][1], self._state[0][0]]])).double()  # 白黒反転させたstateの算出
            # action = self.opposite_net(state_tmp).sample()
            # print("net shape {}".format(self.opposite_net(state_tmp).shape))
            probs = F.softmax(self.opposite_net(state_tmp), dim=1)
            # print("probs {}".format(probs))
            action = select_legal_hand(probs, self.othello.legal_hands(self.opposite_color))  # 行動選択
            self.othello.set_stone_index(action, self.opposite_color)  # 石を置く.
            self._state = self.othello.get_state()  # 状態の更新

    def get_network(self):
        return self.opposite_net

    def step(self, act, show=False):  # coordinateの形でactを指定する
        # state, reward, done を返す.
        if not (act in self.othello.legal_hands(self.your_color, coordinate=True)):
            print("[ERROR] Action you selected is not legal hand.")
            return 0, 0.0, True
        self.othello.set_stone_index(act, self.your_color)  # act を実行
        if show:
            self.othello.print_board()
        self._state = self.othello.get_state()  # state 更新
        flag, winner = self.othello.is_end(self.opposite_color)
        if flag:
            return self._state, (1.0 if winner == self.your_color else -1.0), True
        if self.opposite_color == "white":
            # print("state in  step {}".format(self._state.shape))
            state_tmp = torch.from_numpy(np.array([[self._state[0][1], self._state[0][0]]])).double()  # 白黒反転させたstateの算出
        else:
            state_tmp = torch.from_numpy(self._state).double()

        # print("state tmp {}".format(state_tmp.shape))
        if show:
            self.othello.print_board()  # print board

        probs = self.softmax(self.opposite_net(state_tmp))
        action = select_legal_hand(probs, self.othello.legal_hands(self.opposite_color))  # 行動選択
        self.othello.set_stone_index(action, self.opposite_color)  # 石を置く.
        if show:
            self.othello.print_board()

        self._state = self.othello.get_state()  # 状態の更新
        flag, winner = self.othello.is_end(self.your_color)  # 終了判定
        if flag:
            return self._state, (1.0 if winner == self.your_color else -1.0), True
        else:
            return self._state, 0.0, False

    def get_state(self):
        return self._state

    def import_othello(self, othello_):
        self.othello = othello_
        self._state = self.othello.get_state()
