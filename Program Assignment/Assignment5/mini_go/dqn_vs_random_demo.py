from absl import logging, flags, app
from environment.GoEnv import Go
import time, os
import numpy as np
from algorimths.dqn import DQN
import tensorflow as tf
#使用TensorFlow实现的深度Q网络（DQN）智能体来训练和评估围棋（Go）游戏
#导入必要的模块和类。
# 使用absl库定义了一些命令行参数，这些参数可以在运行脚本时设
FLAGS = flags.FLAGS#初始化命令行参数对象
#定义了一系列命令行参数，设置了默认值和帮助信息
flags.DEFINE_integer("num_train_episodes", 10000,
                     "Number of training episodes for each base policy.")  # 训练回合数。
flags.DEFINE_integer("num_eval", 1000,
                     "Number of evaluation episodes") #评估回合数
flags.DEFINE_integer("eval_every", 2000,
                     "Episode frequency at which the agents are evaluated.") #评估频率。
flags.DEFINE_integer("learn_every", 128,
                     "Episode frequency at which the agents learn.")  #学习频率
flags.DEFINE_integer("save_every", 2000,
                     "Episode frequency at which the agents save the policies.")  #保存模型频率
flags.DEFINE_list("hidden_layers_sizes", [
    128, 128
], "Number of hidden units in the Q-net.")   #Q网络隐藏层大小
flags.DEFINE_integer("replay_buffer_capacity", int(5e4),
                     "Size of the replay buffer.")  #回放缓冲区容量
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")  #储备缓冲区容量


file_path_dqn="saved_model/dqn_vs_random_model"
#main函数是程序的入口点，它初始化环境、智能体，并开始训练和评估过程
def main(unused_argv):
    begin = time.time()#记录程序开始的时间
    env = Go()#创建围棋环境实例
    info_state_size = env.state_size #获取环境的状态大小
    num_actions = env.action_size #获取环境的动作空间大小

    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]  #将命令行参数中的隐藏层大小转换为整数列表
    kwargs = {
        "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
        "epsilon_decay_duration": int(0.6*FLAGS.num_train_episodes),
        "epsilon_start": 0.8,
        "epsilon_end": 0.001,
        "learning_rate": 1e-3,
        "learn_every": FLAGS.learn_every,
        "batch_size": 128,
        "max_global_gradient_norm": 10,
    } #创建一个字典kwargs，包含DQN智能体的参数。
    import agent.agent as agent # 导入agent模块，包含智能体类的定义
    ret = [0]  #初始化一个列表ret，用于记录奖励。
    max_len = 2000 #设置记录奖励列表的最大长度

    with tf.Session() as sess:  #创建一个TensorFlow会话。
        # agents = [DQN(sess, _idx, info_state_size,
        #                   num_actions, hidden_layers_sizes, **kwargs) for _idx in range(2)]  # for self play  创建智能体列表，包含两个相同的DQN智能体这个是用于自我训练的
        agents = [DQN(sess, 0, info_state_size,
                          num_actions, hidden_layers_sizes, **kwargs), agent.RandomAgent(1)]  #创建一个智能体列表，包含一个DQN智能体和一个随机智能体
        sess.run(tf.global_variables_initializer())  #初始化TensorFlow全局变量

        # train the agent 训练智能体
        # 开始训练循环，对于每个训练回合，重置环境，智能体交替行动，直到回合结束，然后更新智能体的状态。
        print(f"训练回合episode数为:{FLAGS.num_train_episodes}")
        for ep in range(FLAGS.num_train_episodes):
            # 评估和保存模型
            if (ep + 1) % FLAGS.eval_every == 0:
                losses = agents[0].loss
                logging.info("Episodes: {}: Losses: {}, Rewards: {}".format(ep + 1, losses, np.mean(ret)))
                # with open('log_{}_{}'.format(os.environ.get('BOARD_SIZE'), begin), 'a+') as log_file:
                with open('log_for_all_models/log_dqn_vs_random/log_{}_{}'.format(os.environ.get('BOARD_SIZE'), begin), 'a+') as log_file:
                    log_file.writelines("{}, {}\n".format(ep+1, np.mean(ret)))
            if (ep + 1) % FLAGS.save_every == 0:
                # 保存模型
                # if not os.path.exists("saved_model"):
                #     os.mkdir('saved_model')
                # agents[0].save(checkpoint_root='saved_model', checkpoint_name='{}'.format(ep+1))
                if not os.path.exists(f"{file_path_dqn}"):
                    os.mkdir(f'{file_path_dqn}')
                agents[0].save(checkpoint_root=f'{file_path_dqn}', checkpoint_name='{}'.format(ep+1))
            time_step = env.reset()  # a go.Position object
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = agent_output.action
                time_step = env.step(action_list)
            for agent in agents:
                agent.step(time_step)
            # 更新奖励记录
            if len(ret) < max_len:
                ret.append(time_step.rewards[0])
            else:
                ret[ep % max_len] = time_step.rewards[0]

        # evaluated the trained agent 评估训练后的智能体
        agents[0].restore(f"{file_path_dqn}/10000")  #加载训练好的模型
        ret = []
        for ep in range(FLAGS.num_eval):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == 0:
                    agent_output = agents[player_id].step(time_step, is_evaluation=True, add_transition_record=False)
                else:
                    agent_output = agents[player_id].step(time_step)
                action_list = agent_output.action
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            # for agent in agents:
            agents[0].step(time_step, is_evaluation=True, add_transition_record=False)
            agents[1].step(time_step)
            ret.append(time_step.rewards[0])
        print(np.mean(ret))

    print('Time elapsed:', time.time()-begin)


if __name__ == '__main__':
    app.run(main)
