import os  #导入了os模块来设置环境变量BOARD_SIZE，用于定义棋盘的大小。
os.environ['BOARD_SIZE'] = '5'
from environment import go, coords
import collections, enum  #导入了collections和enum模块，以及numpy库。
import numpy as np


class TimeStep(#TimeStep是一个命名元组，包含observations（观察结果）、rewards（奖励）、discounts（折扣）和step_type（步骤类型）。
    collections.namedtuple(
        "TimeStep", ["observations", "rewards", "discounts", "step_type"])):
    """Returned with every call to `step` and `reset`.
#TimeStep用于表示游戏的每个步骤的输出，包含当前的观察结果、奖励、折扣和步骤类型。
    A `TimeStep` contains the data emitted by a game at each step of interaction.
    A `TimeStep` holds an `observation` (list of dicts, one per player),
    associated lists of `rewards`, `discounts` and a `step_type`.
first、mid和last方法用于检查当前步骤是否是序列的第一个、中间或最后一个步骤。
    The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
    `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
    have `StepType.MID.
current_player方法返回当前玩家的索引。

    Attributes:
      observations: a list of dicts containing observations per player.
      rewards: A list of scalars (one per player), or `None` if `step_type` is
        `StepType.FIRST`, i.e. at the start of a sequence.
      discounts: A list of discount values in the range `[0, 1]` (one per player),
        or `None` if `step_type` is `StepType.FIRST`.
      step_type: A `StepType` enum value.
    """
    __slots__ = ()

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def current_player(self):
        return self.observations["current_player"]


class StepType(enum.Enum):
    """Defines the status of a `TimeStep` within a sequence."""
#StepType是一个枚举类型，定义了TimeStep在序列中的状态：FIRST（序列的第一个步骤）、MID（序列中的中间步骤）和LAST（序列的最后一个步骤）。
# first、mid和last方法用于检查当前步骤类型。
    FIRST = 0  # Denotes the first `TimeStep` in a sequence.
    MID = 1  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    LAST = 2  # Denotes the last `TimeStep` in a sequence.

    def first(self):
        return self is StepType.FIRST

    def mid(self):
        return self is StepType.MID

    def last(self):
        return self is StepType.LAST


class Go(object):
    def __init__(self, flatten_board_state=True, discount_factor=1.0):
        self.__state = go.Position(komi=0.5)
        self.__flatten_state = flatten_board_state
        self.__discount_factor = discount_factor
        N = int(os.environ.get("BOARD_SIZE"))
        self.__state_size = N ** 2
        self.__action_size = self.__state_size + 1  # board size and an extra action for "pass"
        self.__num_players = 2

    # Go类表示围棋游戏环境。
    # __init__方法初始化游戏状态，包括棋盘大小、动作大小、当前玩家等。
    # state_size和action_size属性分别返回棋盘状态的大小和可能动作的总数。
    # to_play属性返回当前应该行动的玩家。
    # info_state属性返回当前棋盘状态的表示。
    # step方法执行一个动作，并返回下一个状态、奖励、折扣和步骤类型。
    # reset方法重置游戏环境到初始状态，并返回初始状态。
    # get_all_legal_moves方法返回所有合法的动作。
    # get_current_board方法返回当前棋盘状态。
    @property
    def state_size(self):
        return self.__state_size

    @property
    def action_size(self):
        return self.__action_size

    @property
    def to_play(self):
        if self.__state.to_play == 1:  # BLACK (player 1)
            return 0
        else:  # -1 for WHITE (player 2)
            return 1

    @property
    def info_state(self):
        return np.add(self.__state.board, 1)

    # step方法中，玩家执行动作后，更新棋盘状态，并返回新的观察结果、奖励等。
    # reset方法用于开始新游戏，重置棋盘状态。
    # 游戏的结束条件是通过__state.is_game_over()方法判断的。
    def step(self, action):
        """
        In step function, the game of go proceeds with the action taken by the current player and returns a next tuple to the player who is to act next step

        :param action: a place to move for current player
        :return: return a tuple of (next_state, done, reward, info), where the reward for Black (the first player) is 1, -1 and 0.
        """
        # if action not in go.Position.all_legal_moves():  # the go engine will raise an IllegalMove error
        #     raise('Illegal move!')
        #     exit(1)
        # self.state.play_move(action)
        move = coords.from_flat(action)
        self.__state.play_move(move, mutate=True)
        observations = {"info_state": [], "legal_actions": [], "current_player": []}
        for i in range(2):
            # if self.to_play == i:
            if self.__flatten_state:
                _state = np.reshape(self.info_state, (self.__state_size,))
            else:
                _state = self.info_state
            observations["info_state"].append(_state)
            observations['legal_actions'].append(np.where(self.__state.all_legal_moves() == 1)[0])
            # else:
            #     observations["info_state"].append(None)
            #     observations['legal_actions'].append(None)
        observations['current_player'] = self.to_play
        if self.__state.is_game_over():
            return TimeStep(observations=observations, rewards=[self.__state.result(), -self.__state.result()],
                            discounts=[self.__discount_factor] * self.__num_players, step_type=StepType.LAST)
        else:
            return TimeStep(observations=observations, rewards=[0.0, 0.0],
                            discounts=[self.__discount_factor] * self.__num_players, step_type=StepType.MID)

    def reset(self):
        """
        reset the game at the beginning of the game to get an initial state
        :return: should reset the env and return a initial state
        """
        self.__state = go.Position(komi=0.5)
        if self.__flatten_state:
            _state = np.reshape(self.info_state, (self.__state_size,))
        else:
            _state = self.info_state
        observations = {"info_state": [_state, None],
                        "legal_actions": [np.where(self.__state.all_legal_moves() == 1)[0], None],
                        "current_player": self.to_play}
        return TimeStep(observations=observations, rewards=[0.0, 0.0],
                        discounts=[self.__discount_factor] * self.__num_players, step_type=StepType.FIRST)

    def get_all_legal_moves(self):
        return self.__state.all_legal_moves()

    def get_current_board(self):
        return self.__state
