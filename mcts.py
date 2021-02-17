from dataclasses import dataclass
import numpy as np
from Policy_train import PolicyNetwork, transform_state
from env import select_legal_hand
import math
import torch
import torch.nn.functional as F
from my_REINFORCE import REINFORCE
from board import Othello
from tqdm import tqdm


@dataclass
class StateAction:
    state: np.array
    action: int


def step_othello(othello, action, color, show=False):
    if not (action in othello.legal_hands(color, coordinate=True)):
        print("[ERROR] Action you selected is not legal hand. in step_othello function.")
        return othello, 0.0, True
    opposite_color = "white" if color == "black" else "black"
    othello.set_stone_index(action, color)
    othello.print_board(show)
    flag, winner = othello.is_end(opposite_color)  # 終了判定
    if flag:
        return othello, (1.0 if winner == "black" else -1.0), True
    else:
        return othello, 0.0, False


class Node:
    def __init__(self, state, action):  # state : np.zeros((1, 2, 8, 8))
        self._state = state  # state of this node.
        self._action = action  # action of this node.
        self.q_value = 0.0  # Q(state, action)
        self.is_leaf = True  # Is this node a leaf node?
        self.visited_times = 0  # visited times of this node.


class simple_agent:
    def __init__(self, network, color):
        self._color = color
        self._network = network

    def select(self, state, legal_hands):  # stateは特に変換する必要は無い.
        state = torch.from_numpy(transform_state(state, your_color=self._color)).double()  # stateを変換
        probs = F.softmax(self._network(state), dim=1)
        act = select_legal_hand(probs, legal_hands)
        return act



class MCTS_tree:
    def __init__(self, your_color="black", gamma=0.99, rho=0.1):  # your_color : whiteは未完成
        # hyper parameters
        self.rho = rho
        self.gamma = gamma
        self.sikiiti = 10  # この回数以上の経験回数で,葉nodeから子nodeを展開する.(nodes に入れる)
        self.rollout_times = 10  # rolloutを行う回数

        # static values
        self.your_color = your_color  # static
        self.color = "white" if your_color == "black" else "black"  # MCTS側のシミュレートするplayerの色 : static
        self.show = False
        opposite_net = PolicyNetwork().double()  # 対戦相手側の動作をするnetwork : static
        player_net = PolicyNetwork().double()  # player側の動作をするnetwork : static
        opposite_net.load_state_dict(torch.load("./SLnet_trained.pth"))
        player_net.load_state_dict(torch.load("./SLnet_trained.pth"))

        # dynamic values
        # self.agent = REINFORCE(player_net, torch.optim.Adam(player_net.parameters()))  # player側の動作をするnetwork
        # self.agent.training = False
        if self.your_color == "black":
            self.black_agent = simple_agent(player_net, self.your_color)
            self.white_agent = simple_agent(opposite_net, self.color)
        else:
            self.white_agent = simple_agent(player_net, self.your_color)
            self.black_agent = simple_agent(opposite_net, self.color)
        self.othello = Othello()
        self.nodes = {}  # (state, action) -> node の dict型
        legal_hands = self.othello.legal_hands("black", True)
        for act in legal_hands:
            node = Node(self.othello.get_state(), act)
            self.nodes[(self.othello.get_state().tobytes(), act)] = node

    def set_player_network(self, player_net):
        if self.your_color == "black":
            self.black_agent = simple_agent(player_net, self.your_color)
        else:
            self.white_agent = simple_agent(player_net, self.your_color)

    def set_opposite_network(self, opposite_net):
        if self.your_color == "black":
            self.white_agent = simple_agent(opposite_net, self.color)
        else:
            self.black_agent = simple_agent(opposite_net, self.color)

    def _evaluate_node(self, state, action, total_sim_times, color):  # evaluate node
        # total_sim_times = \sum_{a} m(state, a)
        # calc : q(s,a) + \rho \sqrt{ (\sum_{a'} m(s,a')) / m(s,a) }
        if not ((state.tobytes(), action) in self.nodes):
            ans = math.sqrt(float(total_sim_times) / 1.0) * self.rho
        else:
            present_node = self.nodes[(state.tobytes(), action)]
            if present_node.visited_times == 0:
                ans = math.sqrt(float(total_sim_times) / float(present_node.visited_times + 1.0)) * self.rho
            else:
                ans = math.sqrt(float(total_sim_times) / float(present_node.visited_times)) * self.rho
            if color == "black":
                ans += present_node.q_value
            else:
                ans -= present_node.q_value  # whiteの時もmaxで統一したい.
        return ans

    def selection(self, color):  # 現在のstateについて,evaluate_nodeを用いて評価して,maxとなる行動を選択するpolicy
        legal_hands = self.othello.legal_hands(color)
        if len(legal_hands) == 0:
            print("[Error] can't select action in MCTS.selection")
            return 0
        else:
            state = self.othello.get_state()
            total_sim_times = 0.0
            for act in legal_hands:
                if (state.tobytes(), act) in self.nodes:
                    node = self.nodes[(state.tobytes(), act)]
                    total_sim_times += node.visited_times
            mx = self._evaluate_node(state, legal_hands[0], total_sim_times, color)
            ans = legal_hands[0]
            for act in legal_hands:
                if mx < self._evaluate_node(state, act, total_sim_times, color):
                    mx = self._evaluate_node(state, act, total_sim_times, color)
                    ans = act
            return ans  # select action by _evaluate_node

    def _expansion(self, node, opposite_color):  # expansion leaf nodes from "node". opposite_colorはnodeの反対の色
        # これを実行するときには, othello自体は, nodeの次のstateに遷移している事を前提にしている.
        if not node.is_leaf:
            print("[Error] this node is not leaf node. in MCTS._expansion")
        state = self.othello.get_state()  # nodeのstateの次のstate
        legal_hands = self.othello.legal_hands(opposite_color, True)  # n_stateについての合法手
        for act in legal_hands:  # expansion
            new_node = Node(state, act)
            self.nodes[(state.tobytes(), act)] = new_node
        node.is_leaf = False
        return node

    def search(self, color):  # 石がcolorのplayerの現在のstateについての(state, action)を self.sikiiti回 探索する.
        legal_hands = self.othello.legal_hands(color, True)  # legal hands
        for _ in tqdm(range(self.sikiiti)):
            act = self.selection("black")
            self._search(act, color)
            # for action in legal_hands:
            #     self._search(action, color)

    def _search(self, action, color):  # verified
        state = self.othello.get_state()
        if not ((state.tobytes(), action) in self.nodes):  # error
            print("[Error] present (state, action) is not node. in MCTS._search")

        node = self.nodes[(state.tobytes(), action)]  # present node
        opposite_color = "white" if color == "black" else "black"
        self.othello.set_stone_index(action, color)  # act. othello自体はn_stateに遷移している. nodeとずれてる事に注意.

        # evaluate node
        if node.is_leaf:  # present node is a leaf node.
            ans = 0.0
            for _ in range(self.rollout_times):
                ans += float(self.rollout(opposite_color))
            rew = ans/float(self.rollout_times)  # rollout and evaluate leaf node, and back propagate the evaluation.
        else:  # present node is not leaf node.
            act = self.selection(color)  # select next node.
            rew = self._search(act)

        # update
        node.visited_times += 1  # update visited times
        node.q_value += float((rew - node.q_value)) / float(node.visited_times)  # update q value
        if node.is_leaf and (node.visited_times > self.sikiiti):  # expansion
            node = self._expansion(node, opposite_color)
        self.nodes[(state.tobytes(), action)] = node
        self.othello.undo()
        return rew * self.gamma

    def rollout(self, color):  # play game until the episode reaches the end, and undo each steps.
        # action = self.agent.act(input_state[0])
        legal_hands = self.othello.legal_hands(color, True)
        if color == "black":
            action = self.black_agent.select(self.othello.get_state(), legal_hands)
        else:
            action = self.white_agent.select(self.othello.get_state(), legal_hands)
        self.othello, rew, done = step_othello(self.othello, action, color, self.show)

        if done:
            self.othello.undo()
            return rew
        opposite_color = "white" if color == "black" else "black"
        rew = self.rollout(opposite_color)
        self.othello.undo()
        return rew

def main():
    print("test")
    test = MCTS_tree()
    rew = test.rollout("black")
    print("rew is {}".format(rew))
    test.othello.print_board()
    test.search("black")
    legal_hands = test.othello.legal_hands("black")
    for act in legal_hands:
        print(test.nodes[(test.othello.get_state().tobytes(), act)].q_value)

if __name__ == "__main__":
    main()





















