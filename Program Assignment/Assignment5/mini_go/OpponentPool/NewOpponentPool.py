import sys

from networkx.readwrite.nx_yaml import read_yaml
from tensorflow.contrib.learn.python.learn.datasets.base import retry

sys.path.append("../")  #将当前目录的上级目录添加到 Python 模块搜索路径中，以便能够正确导入位于上级目录中的模块。
from absl import logging, flags, app#从 absl 库中导入日志记录（logging）、命令行参数解析（flags）和应用程序运行（app）相关的功能。
from environment.GoEnv import Go#从自定义的 environment.GoEnv 模块中导入 Go 类，推测这个类用于创建和管理围棋游戏环境。
import time, os#导入时间（time）和操作系统（os）相关的模块，用于处理时间相关操作（如记录训练时间）和文件系统操作（如创建目录、读取文件等）。
import numpy as np  #导入 numpy 库并将其别名为 np，numpy 是用于科学计算和数组操作的常用库。
import tensorflow as tf
from algorimths.policy_gradient import PolicyGradient
from algorimths.dqn import DQN#从自定义的 algorimths.dqn 模块中导入 DQN 类，可能是用于实现深度 Q 网络（Deep Q-Network）算法相关的功能
import agent.agent as agent
from OpponentPoolAgent import OpponentPool


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"#设置环境变量，将 TF_CPP_MIN_LOG_LEVEL 设置为 3，用于控制 TensorFlow 的日志级别，减少不必要的日志输出
SELF_AGENT = 0#定义常量，分别表示自己的智能体和对手智能体的标识，方便在代码中区分不同智能体的操作。
RIVAL_AGENT = 1
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"#设置环境变量，将 TF_CPP_MIN_LOG_LEVEL 设置为 3，用于控制 TensorFlow 的日志级别，减少不必要的日志输出
SELF_AGENT = 0#定义常量，分别表示自己的智能体和对手智能体的标识，方便在代码中区分不同智能体的操作。
RIVAL_AGENT = 1
FLAGS = flags.FLAGS#获取命令行参数解析对象，用于后续定义和解析命令行参数。

flags.DEFINE_integer("num_train_episodes", 6000,
                     "Number of training episodes for each base policy.")#定义一个整数类型的命令行参数 num_train_episodes，默认值为 200000，表示每个基础策略的训练 episodes 数量。
flags.DEFINE_integer("num_eval", 1000,
                     "Number of evaluation episodes")
flags.DEFINE_integer("eval_every", 2000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("learn_every", 128,
                     "Episode frequency at which the agents learn.")
flags.DEFINE_integer("save_every", 2000,
                     "Episode frequency at which the agents save the policies.")
flags.DEFINE_list("output_channels", [
    2, 4, 8, 16, 32
], "")
flags.DEFINE_list("hidden_layers_sizes", [
    32, 64, 14
], "Number of hidden units in the net.")
flags.DEFINE_integer("replay_buffer_capacity", int(5e4),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_bool("use_dqn", False, "use dqn or not. If set to false, use a2c")
flags.DEFINE_integer("num_iterations", 200000,
                     "Number of iteration for Opponent Pool")


def use_dqn():
    # return FLAGS.use_dqn
    return False  # 这里人为调整了一下

def init_env():
    '''初始化游戏环境函数'''
    begin = time.time()
    env = Go(flatten_board_state=False)
    info_state_size = env.state_size
    print(info_state_size)
    num_actions = env.action_size
    return env, info_state_size, num_actions, begin

def init_kwargs():
    '''初始化kwargs类型的参数'''
    # if use_dqn():
    dqn_kwargs = {
            "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
            "epsilon_decay_duration": int(0.6 * FLAGS.num_train_episodes),
            "epsilon_start": 0.8,
            "epsilon_end": 0.001,
            "learning_rate": 2e-4,
            "learn_every": FLAGS.learn_every,
            "batch_size": 256,
            "max_global_gradient_norm": 10,
        }
    # else:
    a2c_kwargs = {
            "pi_learning_rate": 3e-4,
            "critic_learning_rate": 1e-3,
            "batch_size": 128,
            "entropy_cost": 0.5,
            "max_global_gradient_norm": 20,
        }
    ret = [0]
    max_len = 2000
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    return dqn_kwargs,a2c_kwargs,hidden_layers_sizes,ret,max_len



def train_agent(agents, env, ret, max_len, begin,sess):
    '''借鉴a2c_vs_random的做法先进行常规训练逻辑'''
    sess.run(tf.global_variables_initializer())
    print(f"训练回合episode数为:{FLAGS.num_train_episodes}")
    for ep in range(FLAGS.num_train_episodes):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            # print("type:",type(agents[player_id]))
            agent_output = agents[player_id].step(time_step)
            action_list = agent_output.action
            time_step = env.step(action_list)
        print(time_step)
        for agent_ in agents:
            agent_.step(time_step)
        if len(ret) < max_len:
            ret.append(time_step.rewards[0])
        else:
            ret[ep % max_len] = time_step.rewards[0]
    return

def save_agent_model(agent, iteration):
    """
    保存智能体模型
    :param agent: PolicyGradient类实例化的智能体对象
    :param iteration: 当前迭代轮次
    """
    save_dir = "OpponentPool_SavedModel/DQN_OpponentPool" if use_dqn() else "OpponentPool_SavedModel/a2c_OpponentPool"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "agent_model_{}.ckpt".format(iteration))
    agent.save(save_path)
    logging.info("Model saved at {}".format(save_path))

def evaluate_agent(agent, ep,ret):
    """
    评估智能体性能
    :param agent: PolicyGradient类实例化的智能体对象
    :param env: 游戏环境对象
    """
    losses = agent.loss
    logging.info("Episodes: {}: Losses: {}, Rewards: {}".format(ep + 1, losses, np.mean(ret)))
    with open('log_for_all_models/log_opponent_a2c/log_pg_{}'.format(os.environ.get('BOARD_SIZE')), 'a+') as log_file:
        log_file.writelines("{}, {}\n".format(ep + 1, np.mean(ret)))


def main(unused_argv):
    # 初始化围棋游戏环境
    env, info_state_size, num_actions, begin = init_env()
    dqn_kwargs,a2c_kwargs,hidden_layers_sizes,ret,max_len=init_kwargs()
    # 初始化对手池，假设池大小为10
    opponent_pool = OpponentPool(pool_size=10)
    with tf.Session() as sess:
        # 根据命令行参数确定使用的算法（DQN或A2C等），并初始化智能体
        if FLAGS.use_dqn:
            main_agent = DQN(sess, 0, env.state_size, env.action_size, hidden_layers_sizes,**dqn_kwargs)
        else:
            main_agent = PolicyGradient(sess, 0, env.state_size, env.action_size, hidden_layers_sizes,**a2c_kwargs)

        # 训练当前智能体  先按照常规逻辑进行训练
        # agent.train()
        import agent.agent as agent
        agents = [main_agent, agent.RandomAgent(1)]
        for i in range(5):
            train_agent(agents, env, ret, max_len, begin, sess)
            # 将当前迭代后的智能体模型添加到对手池中
            opponent_pool.add_opponent(main_agent)
            print("ok")

        # 进行多轮训练迭代
        for iteration in range(FLAGS.num_iterations):
            # 从对手池中采样几个对手用于后续和当前智能体对抗训练
            sampled_opponents = opponent_pool.sample_opponents(num_opponents=3)
            for opponent in sampled_opponents:
                # 进行对抗训练，在一个模拟环境中让智能体和对手进行交互并根据结果更新智能体
                agents=[main_agent, opponent]
                train_agent(agents, env, ret, max_len, begin, sess)
                # 将当前迭代后的智能体模型添加到对手池中
                opponent_pool.add_opponent(main_agent)
                # 定期评估智能体性能并保存模型（根据eval_every和save_every参数）
            if (iteration + 1) % FLAGS.eval_every == 0:
                evaluate_agent(main_agent, iteration,ret)
            if (iteration + 1) % FLAGS.save_every == 0:
                save_agent_model(main_agent, iteration)



if __name__ == '__main__':
    app.run(main)


