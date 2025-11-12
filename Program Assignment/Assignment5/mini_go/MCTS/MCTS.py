import numpy as np
from operator import itemgetter
import copy


class TreeNode(object):
    '''TreeNode 类表示蒙特卡洛树中的一个节点。
    方法：
        expand：扩展节点，为每个可能的动作创建子节点。
        select：选择具有最高价值的子节点进行探索。
        update：更新节点的统计数据。
        update_recursive：递归更新从当前节点到根节点的所有节点的统计数据。
        get_value：计算节点的总价值，用于选择操作。
        is_leaf：检查节点是否为叶节点。
        is_root：检查节点是否为根节点。
    '''
    def __init__(self, parent, prior_p):
        '''
        初始化成员：
            parent：父节点。
            prior_p：先验概率。
            _children：子节点字典，键为动作，值为对应的 TreeNode。
            _n_visits：节点被访问的次数。
            _Q：平均行动价值。
            _u：探索因子。
            _P：先验概率。'''
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = prior_p
        self._P = prior_p
    def expand(self, action_priors):
        '''扩展节点，接受一个包含动作和对应先验概率的列表 action_priors。对于列表中的每个动作和概率，如果该动作对应的子节点不存在，
        则创建一个新的 TreeNode 作为子节点，并将其添加到 _children 字典中。新子节点的父节点为当前节点，先验概率为传入的概率值。'''
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self):
        '''选择当前节点的子节点进行探索。它遍历当前节点的所有子节点，通过调用每个子节点的 get_value 方法获取其价值，
        并返回价值最高的子节点。这里使用 max 函数和一个匿名函数来实现。'''
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value())

    def update(self, leaf_value, c_puct):
        '''更新节点的统计信息。首先，将节点的访问次数 _n_visits 加 1。然后，根据新的评估值 leaf_value 更新平均行动价值 _Q，
        使用增量平均的方式更新，以避免每次重新计算所有访问的平均值。如果当前节点不是根节点，还会根据 c_puct（控制探索与利用的平衡参数）、
        父节点的访问次数 _parent._n_visits 和当前节点的先验概率 _P 更新探索因子 _u。'''
        self._n_visits += 1

        self._Q += (leaf_value - self._Q) / self._n_visits

        if not self.is_root():
            self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)

    def update_recursive(self, leaf_value, c_puct):
        '''递归更新  从当前节点开始，一直向上更新到根节点。如果当前节点有父节点，则先递归调用父节点的 update_recursive 方法，
        然后再更新当前节点自身。这样可以确保在一次模拟结束后，从叶节点到根节点的路径上的所有节点都能得到正确的更新。'''
        if self._parent:
            self._parent.update_recursive(leaf_value, c_puct)
        self.update(leaf_value, c_puct)

    def get_value(self):
        '''计算节点的总价值，用于在节点选择过程中比较不同子节点的优劣。总价值由平均行动价值 _Q 和探索因子 _u 相加得到。'''
        return self._Q + self._u

    def is_leaf(self):
        """判断当前节点是否为叶节点，即是否没有子节点。如果 _children 字典为空，则表示当前节点是叶节点。"""
        return self._children == {}

    def is_root(self):
        '''判断当前节点是否为根节点，即是否没有父节点。如果 _parent 为 None，则表示当前节点是根节点。'''
        return self._parent is None


class MCTS(object):
    '''MCTS类实现了Ment Carl树的搜索算法
        _exchange_player：交换当前玩家。
        _playout：执行一次模拟，从根节点到叶节点，然后评估叶节点，并更新路径上的所有节点。
        _evaluate_rollout：使用展开策略从当前状态执行到游戏结束，并返回结果。
        get_move：执行多次模拟，并返回最常访问的动作。
        update_with_move：根据上一次选择的动作更新根节点。
    '''
    def __init__(self, value_fn, policy_fn, rollout_policy_fn, lmbda=0.5, c_puct=5,
                 rollout_limit=100, playout_depth=10, n_playout=100):
        '''初始化方法：
            value_fn：评估函数，用于评估当前状态的价值。
            policy_fn：策略函数，用于选择动作。
            rollout_policy_fn：展开策略函数，用于从叶节点展开到游戏结束。
            其他参数如 c_puct（控制探索与利用的平衡），rollout_limit（展开的步数限制），n_playout（模拟次数）'''
        self._root = TreeNode(None, 1.0)
        self._value = value_fn
        self._policy = policy_fn
        self._rollout = rollout_policy_fn
        self._lmbda = lmbda
        self._c_puct = c_puct
        self._rollout_limit = rollout_limit
        self._L = playout_depth
        self._n_playout = n_playout
        self._player_id = 0
        # self._env = env

    def _exchange_player(self):
        '''交换玩家  如果当前玩家为 0，则将其设置为 1；如果当前玩家为 1，则将其设置为 0。在模拟过程中，用于交替玩家进行决策。'''
        self._player_id = 1 if self._player_id == 0 else 0

    def _playout(self, state, env, leaf_depth):
        '''执行一次蒙特卡洛树的模拟过程。从根节点开始，在循环中，如果当前节点是叶节点，
        则使用策略函数 self._policy 获取动作概率分布 action_probs，并扩展节点。
        然后选择具有最高价值的子节点进行探索，执行选择的动作 action，通过环境的 step 方法更新状态 state，并交换玩家。
        当达到模拟深度 leaf_depth 或者遇到无法扩展的节点（动作概率列表为空）时停止循环。
        最后，根据 lmbda 的值，结合价值函数评估值 v 和模拟结果评估值 z 计算叶节点的价值 leaf_value，
        并递归更新从叶节点到根节点路径上的所有节点'''
        #初始化时，创建一个根节点。
        node = self._root
        for i in range(leaf_depth):
            # Only expand node if it has not already been done. Existing nodes already know their
            # prior. 在 _playout 方法中，如果遇到叶节点，则使用策略函数扩展节点；否则，选择具有最高价值的子节点进行探索。
            if node.is_leaf():
                action_probs = self._policy(state, self._player_id)
                if len(action_probs) == 0:
                    break
                node.expand(action_probs)
            # Greedily select next move.
            action, node = node.select()  #在每次模拟结束时，使用评估函数和展开策
            # state.do_move(action)
            state = env.step(action)  #最后，选择最常访问的子节点作为下一步的行动
            self._exchange_player()
        v = self._value(state, self._player_id) if self._lmbda < 1 else 0
        z = self._evaluate_rollout(state, env, self._rollout_limit) if self._lmbda > 0 else 0
        leaf_value = (1 - self._lmbda) * v + self._lmbda * z

        node.update_recursive(leaf_value, self._c_puct)

    def _evaluate_rollout(self, state, env, limit):
        """使用展开策略 self._rollout 从给定的状态 state 开始进行模拟，直到游戏结束或达到步数限制 limit。在循环中，
        根据展开策略选择动作 max_action，执行动作并更新状态，交换玩家。如果状态表示游戏已经结束（state.last() 为真），
        则跳出循环。最后返回游戏的奖励，用于评估模拟结果。
        """
        player = state.observations["current_player"]
        for i in range(limit):
            action_probs = self._rollout(state, self._player_id)
            # if len(action_probs) == 0:
            #     break
            max_action = max(action_probs, key=itemgetter(1))[0]
            # state.do_move(max_action)
            state = env.step(max_action)
            self._exchange_player()
            if state.last():
                break
        return state.rewards[0]

    def get_move(self, state, env):
        '''在 get_move 方法中，进行多次模拟（_n_playout 次），
        每次模拟从根节点开始，直到达到一定的深度（_L），然后评估叶节点，并更新路径上的所有节点。'''
        self._player_id = 0
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            env_cpy = copy.deepcopy(env)
            self._playout(state_copy, env_cpy, self._L)
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        '''执行一步动作后更新蒙特卡洛树的根节点。如果上一步选择的动作 last_move 在当前根节点的子节点中存在，
        则将根节点更新为对应的子节点，并将新根节点的父节点设置为 None。如果动作不存在，则创建一个新的根节点，
        可能表示进入了一个新的状态，需要重新开始搜索'''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


