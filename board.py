import numpy as np
import copy


class board:
    def __init__(self):
        self.black = np.zeros((8, 8))
        self.white = np.zeros((8, 8))
        self.black[4][3] = 1.0
        self.black[3][4] = 1.0
        self.white[3][3] = 1.0
        self.white[4][4] = 1.0
        # board

        self.black_stone = 2
        self.white_stone = 2
        # 石の数

    def copy(self):
        return copy.deepcopy(self)

    def reset(self):
        self.black = np.zeros((8, 8))
        self.white = np.zeros((8, 8))
        self.black[4][3] = 1.0
        self.black[3][4] = 1.0
        self.white[3][3] = 1.0
        self.white[4][4] = 1.0
        # board

        self.black_stone = 2
        self.white_stone = 2
        # 石の数


class Othello:
    def __init__(self):
        self.board = board()  # 現在のボード
        self._board_stack = []  # 過去のボードの履歴, append / pop で使えばok
        self._delta = [1, 0, -1]  # 計算用

        self._already_make_legal = False
        self._legal_black = []
        self._legal_white = []

    def print_board(self, show=True):
        if not show:
            return
        print(" ", end="|")
        for i in range(8):
            print(i + 1, end="|")
        print("")

        for _ in range(9):
            print("=", end="=")
        print("")

        for i in range(8):
            print(i + 1, end="|")
            for j in range(8):
                b = self.board.black[j][i]
                w = self.board.white[j][i]

                if b == 1 and w == 1:
                    print("[Error] in board.")
                if b == 1 and w != 1:
                    print("x", end="|")
                elif w == 1 and b != 1:
                    print("o", end="|")
                else:
                    print(" ", end="|")
            print("")

            for _ in range(9):
                if i == 7:
                    print("=", end="=")
                else:
                    print("-", end="-")
            print("")

        print(" ", end="|")
        for i in range(8):
            print(i + 1, end="|")
        print("")
        print("black:{}, white:{}".format(self.board.black_stone, self.board.white_stone))
        print("")
        print("")

    def is_stone(self, x, y):  # そこに石はあるんか?
        return self.board.black[x][y] == 1 or self.board.white[x][y] == 1

    def _set_stone(self, x, y, bd_now, bd_opp, dx, dy, flag=1.0):  # flag=1 -> 実際に石を配置する. flag=2 -> 石が置けるか判定する
        # x,y : 座標
        # bd_now : 操作したい色の板,  bd_opp : もう一方の色の板
        # dx, dy : 座標をずらす分
        if x < 0 or x > 7 or y < 0 or y > 7:
            return -1
        if bd_now[x][y] == 1.0:
            return 0  # 見ている色の石に到達した場合
        elif bd_opp[x][y] != 1.0:
            return -1  # そもそも石がない場合
        num = self._set_stone(x + dx, y + dy, bd_now, bd_opp, dx, dy, flag)
        if num >= 0:
            bd_now[x][y] += flag * 1.0
            bd_opp[x][y] += -flag * 1.0
            return num + 1
        else:
            return -1

    def set_stone(self, x, y, color, flag=1.0):  # color = "black"|"white", flag = 1.0 -> 実際に石を配置, 0.0 -> 置けるか判定するだけ
        if self.is_stone(x, y):
            return 0
        num = 0
        bd_now = self.board.black if color == "black" else self.board.white
        bd_opp = self.board.white if color == "black" else self.board.black
        if flag == 1.0:
            old_board = self.board.copy()  # 石を置く前のボード
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                tmp = self._set_stone(x + self._delta[i], y + self._delta[j], bd_now, bd_opp, self._delta[i],
                                      self._delta[j], flag)
                tmp = tmp if tmp > 0 else 0
                num += tmp  # num : ひっくり返った石の数
        if num > 0:
            bd_now[x][y] = 1.0 * flag
            if flag != 1.0:
                return num
            self._already_make_legal = False
            if color == "black":
                self.board.black_stone += num + 1
                self.board.white_stone -= num
            else:
                self.board.white_stone += num + 1
                self.board.black_stone -= num

            self._board_stack.append(old_board)
            return num + 1  # 増えた石の数
        else:
            return 0

    def set_stone_index(self, index, color):  # 11((1,3)に対応)のような座標の指定で石を配置するメソッド
        x, y = index // 8, index % 8
        self.set_stone(x, y, color, 1)

    def can_set(self, x, y, color):  # そこに石は置けるか?
        if self.is_stone(x, y):
            return False
        return True if self.set_stone(x, y, color, 0.0) > 0 else False

    def _make_legal(self):
        if self._already_make_legal:
            return
        self._legal_black = []
        self._legal_white = []
        for i in range(8):
            for j in range(8):
                if self.can_set(i, j, "black"):
                    # self._legal_black.append((i, j))
                    self._legal_black.append(i * 8 + j)
                if self.can_set(i, j, "white"):
                    # self._legal_white.append((i, j))
                    self._legal_white.append(i * 8 + j)
        self._already_make_legal = True

    def legal_hands(self, color, coordinate=False):  # coordinate = True だと,coordinate型で値を返す.
        self._make_legal()
        if coordinate:
            return self._legal_black.copy() if color == "black" else self._legal_white.copy()
        else:
            # if color == "black":
            #     ans = [(x//8, x % 8) for x in enumerate(self._legal_black)]
            # else:
            #     ans = [(x//8, x % 8) for x in enumerate(self._legal_white)]
            # return ans
            return self._legal_black if color == "black" else self._legal_white

    def undo(self):
        self.board = self._board_stack.pop()
        self._already_make_legal = False

    def get_state(self):
        return np.array((self.board.black, self.board.white)).reshape(1, 2, 8, 8)

    def reset(self):
        self.board.reset()
        self._board_stack = []
        self._already_make_legal = False

    def is_end(self, color="black"):  # True/False, color("black", "white", "none") を返す. 引数の color は次に打つプレイヤーを意味する
        if self.board.black_stone + self.board.white_stone == 64:  # 盤が埋まった時
            if self.board.black_stone > self.board.white_stone:
                return True, "black"
            elif self.board.white_stone > self.board.black_stone:
                return True, "white"
        # 合法てが無くなった時
        legal_hands = self.legal_hands(color)
        if len(legal_hands) == 0:
            # print("is_end: {}".format(color))
            # return True, ("white" if color == "black" else "black")
            return True, ("white" if self.board.black_stone < self.board.white_stone else "black")
        else:
            return False, "none"


def check_set_stone(state, color, x, y):  # (x, y)にcolorの石を置けるか判定する. state : np.array (1,2,8,8)
    # print("state's shape0 {}".format(state.shape))
    if color == "black":
        bd_now = state[0][0]
        bd_opp = state[0][1]
    else:
        bd_now = state[0][1]
        bd_opp = state[0][0]
    if (bd_now[x][y] != 0.0) or (bd_opp[x][y] != 0.0):  # もう石がおいてある.
        return False
    for i in range(3):
        for j in range(3):
            dx, dy = i - 1, j - 1
            if (dx == 0) and (dy == 0):
                continue
            tmp = check_set_stone_internal(x + dx, y + dy, bd_now, bd_opp, dx, dy)
            if tmp > 0:
                return True
    return False


def check_set_stone_internal(x, y, bd_now, bd_opp, dx, dy):  # 石が置けるか判定する
    # x,y : 座標
    # bd_now : 操作したい色の板,  bd_opp : もう一方の色の板
    # dx, dy : 座標をずらす分
    if x < 0 or x > 7 or y < 0 or y > 7:
        return -1
    if bd_now[x][y] == 1.0:
        return 0  # 見ている色の石に到達した場合
    elif bd_opp[x][y] != 1.0:
        return -1  # そもそも石がない場合
    num = check_set_stone_internal(x + dx, y + dy, bd_now, bd_opp, dx, dy)
    if num >= 0:
        return num + 1
    else:
        return -1


def check_set_stone_matrix(state, color):  # 置ける場所は 1.0, おけない場所は 0.0 が入る配列を返す.
    return [1.0 if check_set_stone(state, color, i // 8, i % 8) else 0.0 for i in range(64)]


def main():
    print("test")
    tmp = Othello()
    print(tmp.get_state())




if __name__ == "__main__":
    main()
