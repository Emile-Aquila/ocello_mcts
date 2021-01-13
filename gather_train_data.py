from model import PolicyNetwork
from board import othello
import torch
from env import Environment, select_legal_hand
import numpy as np
from dataclasses import dataclass


@dataclass
class train_data:
    state: np.array  # (8, 8)
    color: str  # black / white / none # 勝った方の色を保存, noneはあいこ


data_queue = np.array([])  # ここに対戦データを保存する.
policy_try_times = 1  # policy net同士で対戦して,勝敗を決める事を繰り返す回数


def play_before_step_n(p_net, p_net_op, sl_net, sl_net_op, n):  # N回まではSL networkで. それ以降はPolicy netでプレイする.
    # N回目の黒の盤面情報を保存するが,それ以降はPolicy net同士で対戦して,勝敗を決める.
    env = Environment(sl_net_op, "black")
    flag = False
    for i in range(n):
        probs = sl_net(torch.from_numpy(env.get_state()).double())
        act = select_legal_hand(probs, env.othello.legal_hands("black"))
        _, _, done = env.step(act)
        if done:
            return False
    state = env.get_state()  # N回目の試行が終わった段階でのstateを保存する
    env2 = Environment(p_net_op, "black")
    env2.import_othello(env.othello)  # env2に以降する
    while True:
        probs = p_net(torch.from_numpy(env2.get_state()).double())
        act = select_legal_hand(probs, env2.othello.legal_hands("black"))
        _, rew, done = env.step(act)
        if done:
            if rew == 1.0:
                color = "black"
            elif rew == -1.0:
                color = "white"
            else:
                color = "none"
            ans = train_data(state, color)
            global data_queue
            data_queue = np.append(data_queue, ans)
            return True

