#'''定义了一个简单的强化学习环境中的智能体（Agent）类，以及一个具体的智能体实现——随机智能体（RandomAgent）。'''

import random, collections#导入了Python的标准库中的random模块和collections模块。
# random模块用于生成随机数，而collections模块提供了额外的数据类型，比如这里的namedtuple
StepOutput = collections.namedtuple("step_output", ["action", "probs"])
#定义了一个名为StepOutput的命名元组，它有两个字段：action和probs。命名元组是一种轻量级的对象类型，它提供了不可变的数据结构，并允许通过名称来访问其字段



class Agent(object):
    '''这里定义了一个名为Agent的基类，它有两个方法：构造函数__init__和step方法。
    构造函数目前是空的，而step方法被设计用来在强化学习环境中执行一步操作，
    但在这个基类中，它仅仅抛出一个NotImplementedError异常，
    这意味着任何继承自Agent的子类都需要实现这个方法。'''
    def __init__(self):
        pass

    def step(self, timestep):
        raise NotImplementedError


class RandomAgent(Agent):
    '''这里定义了一个名为RandomAgent的子类，它继承自Agent类。
    它的构造函数接受一个参数_id，这个参数用于标识智能体的ID。
    super().__init__()调用了基类的构造函数。'''
    def __init__(self, _id):
        super().__init__()
        self.player_id = _id

    def step(self, timestep):
        '''在RandomAgent类中，step方法被实现了。这个方法接受一个参数timestep，它通常包含了环境的状态信息。
        方法内部，首先确定当前是哪个玩家在行动（cur_player），然后从当前玩家的合法动作列表（legal_actions）中随机选择一个动作。
        最后，它返回一个StepOutput命名元组，其中包含了所选择的动作和该动作的概率
        （在这里是1.0，因为动作是随机选择的，所以每个合法动作被选中的概率是相同的）。'''
        cur_player = timestep.observations["current_player"]
        return StepOutput(action=random.choice(timestep.observations["legal_actions"][cur_player]), probs=1.0)
