import sys
sys.path.append("../")  #将当前目录的上级目录添加到 Python 模块搜索路径中，以便能够正确导入位于上级目录中的模块。
from absl import logging, flags, app#从 absl 库中导入日志记录（logging）、命令行参数解析（flags）和应用程序运行（app）相关的功能。
from environment.GoEnv import Go#从自定义的 environment.GoEnv 模块中导入 Go 类，推测这个类用于创建和管理围棋游戏环境。
import time, os#导入时间（time）和操作系统（os）相关的模块，用于处理时间相关操作（如记录训练时间）和文件系统操作（如创建目录、读取文件等）。
import numpy as np  #导入 numpy 库并将其别名为 np，numpy 是用于科学计算和数组操作的常用库。
import tensorflow as tf
from algorimths.policy_gradient import PolicyGradient
from algorimths.dqn import DQN#从自定义的 algorimths.dqn 模块中导入 DQN 类，可能是用于实现深度 Q 网络（Deep Q-Network）算法相关的功能
import agent.agent as agent

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"#设置环境变量，将 TF_CPP_MIN_LOG_LEVEL 设置为 3，用于控制 TensorFlow 的日志级别，减少不必要的日志输出
SELF_AGENT = 0#定义常量，分别表示自己的智能体和对手智能体的标识，方便在代码中区分不同智能体的操作。
RIVAL_AGENT = 1
FLAGS = flags.FLAGS#获取命令行参数解析对象，用于后续定义和解析命令行参数。

flags.DEFINE_integer("num_train_episodes", 200000,
                     "Number of training episodes for each base policy.")#定义一个整数类型的命令行参数 num_train_episodes，默认值为 200000，表示每个基础策略的训练 episodes 数量。
flags.DEFINE_integer("num_eval", 1000,
                     "Number of evaluation episodes")
flags.DEFINE_integer("eval_every", 2000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("learn_every", 128,
                     "Episode frequency at which the agents learn.")
flags.DEFINE_integer("save_every", 5000,
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
#####这里先暂时默认是不用dqn算法好了





def get_max_idx(path):
    '''定义获取目录中最大索引值的函数'''
    all_models = []#初始化一个空列表，用于存储目录中的模型文件名（不包含扩展名）。
    for i in list(os.walk(path))[-1][-1]:
        all_models.append(i.split(".")[0])
        #使用 os.walk 遍历指定路径 path 下的所有文件和目录，list(os.walk(path))[-1][-1] 提取出最后一个目录中的所有文件名列表。
    max_idx = max([eval(i) for i in all_models if i.isdigit()])
    return max_idx
def use_dqn():
    '''定义函数判断是否使用 DQN 算法，返回命令行参数中指定的是否使用 DQN 算法的布尔值。'''
    # return FLAGS.use_dqn
    return False  #这里人为调整了一下

###指定之后使用对手池方法之后保存模型的路径
SaveModelPath="OpponentPool_SavedModel/DQN_OpponentPool" if use_dqn() else "OpponentPool_SavedModel/a2c_OpponentPool"

def fmt_hyperparameters():
    '''格式化超参数函数'''
    fmt = ""#初始化一个空字符串，用于构建格式化后的超参数字符串
    for i in FLAGS.output_channels:
        fmt += '_{}'.format(i)
    fmt += '**'
    for i in FLAGS.hidden_layers_sizes:
        fmt += '_{}'.format(i)
    return fmt


def init_env():
    '''初始化游戏环境函数'''
    begin = time.time()
    env = Go(flatten_board_state=False)
    info_state_size = env.state_size
    print(info_state_size)
    num_actions = env.action_size
    return env, info_state_size, num_actions, begin


def init_hyper_paras():
    num_cnn_layer = len(FLAGS.output_channels)
    kernel_shapes = [3 for _ in range(num_cnn_layer)]
    strides = [1 for _ in range(num_cnn_layer)]
    paddings = ["SAME" for _ in range(num_cnn_layer - 1)]
    paddings.append("VALID")
    parameters = [FLAGS.output_channels, kernel_shapes, strides, paddings]
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    if use_dqn():
        kwargs = {
            "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
            "epsilon_decay_duration": int(0.6 * FLAGS.num_train_episodes),
            "epsilon_start": 0.8,
            "epsilon_end": 0.001,
            "learning_rate": 2e-4,
            "learn_every": FLAGS.learn_every,
            "batch_size": 256,
            "max_global_gradient_norm": 10,
        }
    else:
        kwargs = {
            "pi_learning_rate": 3e-4,
            "critic_learning_rate": 1e-3,
            "batch_size": 128,
            "entropy_cost": 0.5,
            "max_global_gradient_norm": 20,
        }
    ret = [0]
    max_len = 2000
    return parameters, hidden_layers_sizes, kwargs, ret, max_len


def init_agents(sess, info_state_size, num_actions, dqn_parameters, hidden_layers_sizes, rival_path, **kwargs):
    if use_dqn():
        Algorithm = DQN(sess, 0, info_state_size ** 0.5, num_actions,
                        dqn_parameters, hidden_layers_sizes, **kwargs)
    else:
        with tf.name_scope("rival"):
            # rival = PolicyGradient(sess, 1, info_state_size ** 0.5, num_actions,
            #                        dqn_parameters, hidden_layers_sizes, **kwargs)
            rival = PolicyGradient(sess, 1, info_state_size ** 0.5, num_actions,
                                   "a2c",None, hidden_layers_sizes, **kwargs)
            # sess.run(tf.local_variables_initializer())
        with tf.name_scope("self"):
            self_ = PolicyGradient(sess, 0, info_state_size ** 0.5, num_actions,
                                   "a2c",None, hidden_layers_sizes, **kwargs)
            # sess.run(tf.local_variables_initializer())
            # self_.restore(rival_path)

    agents = [self_, rival]
    sess.run(tf.global_variables_initializer())
    ###为什么这里的restore会一直有问题？？？？
    print("rival_path的值为:", rival_path)
    # saver = tf.train.import_meta_graph(rival_path)
    # saver.restore(sess, tf.train.latest_checkpoint(rival_path))
    rival.restore(rival_path)
    print("ok")
    restore_agent_op = tf.group([
        tf.assign(self_v, rival_v)
        for (self_v, rival_v) in zip(self_.session, rival.session)
    ])
    sess.run(restore_agent_op)

    # agents[SELF_AGENT].restore(rival_path)
    # agents[RIVAL_AGENT].restore(rival_path)

    logging.info("Load self and rival agents ok!!")

    return agents


def prt_logs(ep, agents, ret, begin):
    losses = agents[0].loss
    logging.info("Episodes: {}: Losses: {}, Rewards: {}".format(ep + 1, losses, np.mean(ret)))

    # alg_tag = "dqn_cnn_vs_rand" if use_dqn() else "a2c_cnn_vs_rnd"

    # with open('current_logs/log_{}_{}'.format(os.environ.get('BOARD_SIZE'), alg_tag + fmt_hyperparameters()),
    with open('current_logs/log_{}_{}'.format(os.environ.get('BOARD_SIZE'), SaveModelPath + fmt_hyperparameters()),
              'a+') as log_file:
        log_file.writelines("{}, {}\n".format(ep + 1, np.mean(ret)))


def save_model(ep, agents):
    # alg_tag = "current_models/CNN_DQN" if use_dqn() else "current_models/CNN_A2C"

    # if not os.path.exists(alg_tag + fmt_hyperparameters()):
    #     os.mkdir(alg_tag + fmt_hyperparameters())
    if not os.path.exists(SaveModelPath + fmt_hyperparameters()):
        os.mkdir(SaveModelPath + fmt_hyperparameters())
    # agents[0].save(checkpoint_root=alg_tag + fmt_hyperparameters(), checkpoint_name='{}'.format(ep + 1))
    agents[0].save(checkpoint_root=SaveModelPath + fmt_hyperparameters(), checkpoint_name='{}'.format(ep + 1))

    print("Model Saved!")


def restore_model(agents, path=None):
    # alg_tag = "current_models/CNN_DQN" if use_dqn() else "current_models/CNN_A2C"
    try:
        if path:
            agents[0].restore(path)
            idex = path.split("/")[-1]
        else:
            idex = get_max_idx(SaveModelPath + fmt_hyperparameters())
            path = os.path.join(SaveModelPath + fmt_hyperparameters(), str(idex))
            agents[0].restore(path)

        logging.info("Agent model restored at {}".format(path))
    except:
        print(sys.exc_info())
        logging.info("No saved Model!!")
        idex = 0

    return int(idex)


def train(agents, env, ret, max_len, begin):
    logging.info("Train on " + fmt_hyperparameters())
    # global_ep = 0
    global_ep = restore_model(agents)
    # global_ep = restore_model(agents,"./used_model/38000")
    try:
        for ep in range(FLAGS.num_train_episodes):
            if (ep + 1) % FLAGS.eval_every == 0:
                prt_logs(global_ep + ep, agents, ret, begin)
            if (ep + 1) % FLAGS.save_every == 0:
                save_model(global_ep + ep, agents)
            time_step = env.reset()  # a go.Position object
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step, is_rival=(player_id == RIVAL_AGENT))
                action_list = agent_output.action
                # print(player_id)
                # print(action_list)
                time_step = env.step(action_list)
            for agent in agents:
                agent.step(time_step)
            if len(ret) < max_len:
                ret.append(time_step.rewards[0])
            else:
                ret[ep % max_len] = time_step.rewards[0]
    except KeyboardInterrupt:
        save_model(global_ep + ep, agents)


def evaluate(agents, env):
    global_ep = restore_model(agents, "../saved_model/CNN_A2C_2_4_8_16_32**_32_64_14/225000")
    # global_ep = restore_model(agents,"../used_model/125000") # ! Good Model!!! 2,2,4,4,8,16; 32,64,14
    # global_ep = restore_model(agents,"../used_model/160000") # ! Good Model!!! 2,2,4,4,8,16; 32,64,14 winning rate:72%

    ret = []

    for ep in range(FLAGS.num_eval):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            if player_id == 0:
                agent_output = agents[player_id].step(time_step, is_evaluation=True)
            else:
                agent_output = agents[player_id].step(time_step)
            action_list = agent_output.action
            time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        # for agent in agents:
        agents[0].step(time_step, is_evaluation=True)
        agents[1].step(time_step)
        ret.append(time_step.rewards[0])

    return ret


def stat(ret, begin):
    print("平均时间为:",np.mean(ret))

    print('Time elapsed:', time.time() - begin)


def main(unused_argv):

    env, info_state_size, num_actions, begin = init_env()
    parameters, hidden_layers_sizes, kwargs, ret, max_len = init_hyper_paras()
    with tf.Session() as sess:
        # rival_path = "rivals/a2c_0"
        rival_model=None
        if use_dqn():
            rival_path = "../saved_model/dqn_vs_random_model"
        else:
            rival_path = "../saved_model/a2c_vs_random_model"
        # rival_path = "../saved_model/CNN_A2C" + fmt_hyperparameters()
        # model_files = [f for f in os.listdir(rival_path) if f.startswith('model.ckpt')]
        # if model_files:
        #     rival_path = os.path.join(rival_path, model_files[0])
        # else:
        #     raise FileNotFoundError("No model file found in the directory.")
        ####通过os.path.join操作获得最终的模型结果   这就体现出了get_max_idx函数的重要性
        # rival_model = os.path.join(rival_path, str(get_max_idx(rival_path)))
        #####初始化agent
        agents = init_agents(sess, info_state_size, num_actions, parameters, hidden_layers_sizes, rival_path,
                             **kwargs)
        ####训练agent
        train(agents, env, ret, max_len, begin)

        ret = evaluate(agents, env)

        stat(ret, begin)


if __name__ == '__main__':
    app.run(main)




