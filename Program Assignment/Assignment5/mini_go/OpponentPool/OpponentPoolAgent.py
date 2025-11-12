import random
import copy

class OpponentPool:
    def __init__(self, pool_size):
        """
        初始化对手池
        :param pool_size: 对手池的最大容量
        """
        self.pool_size = pool_size
        self.pool = []

    def add_opponent(self, opponent_model):
        """
        将对手模型添加到对手池中
        :param opponent_model: 某次迭代后的模型（这里假设可以直接复制整个模型对象，实际可能需要根据模型具体情况调整复制方式）
        """
        if len(self.pool) < self.pool_size:
            self.pool.append(copy.deepcopy(opponent_model))
        else:
            # 如果池已满，随机替换一个已有的对手
            replace_index = random.randint(0, self.pool_size - 1)
            self.pool[replace_index] = copy.deepcopy(opponent_model)

    def sample_opponents(self, num_opponents):
        """
        从对手池中采样一定数量的对手用于后续训练
        :param num_opponents: 要采样的对手数量
        :return: 采样得到的对手模型列表
        """
        return random.sample(self.pool, min(num_opponents, len(self.pool)))


