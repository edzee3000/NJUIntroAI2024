import sys
sys.path.append("./")
from absl import logging, flags, app
from environment.GoEnv import Go
import time, os
import numpy as np
import tensorflow as tf
from algorimths.policy_gradient import PolicyGradient
from algorimths.dqn import DQN
import agent.agent as agent
from MCTSAgent import MCTSAgent

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", 200000,
                     "Number of training episodes for each base policy.")
flags.DEFINE_integer("num_eval", 1000,
                     "Number of evaluation episodes")
flags.DEFINE_integer("eval_every", 2000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("learn_every", 128,
                     "Episode frequency at which the agents learn.")
flags.DEFINE_integer("save_every", 5000,
                     "Episode frequency at which the agents save the policies.")
flags.DEFINE_list("output_channels",[
    2,4,8,16,32
],"")
flags.DEFINE_list("hidden_layers_sizes", [
    32,64,14
], "Number of hidden units in the net.")
flags.DEFINE_integer("replay_buffer_capacity", int(5e4),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_bool("use_dqn",False,"use dqn or not. If set to false, use a2c")
flags.DEFINE_float("lr",2e-4,"lr")
flags.DEFINE_integer("pd",10, "playout_depth")
flags.DEFINE_integer("np",100, "n_playout")


file_path_a2c="saved_model/mcts_vs_a2c_model"
def main(unused_argv):
    begin = time.time()
    env = Go()
    info_state_size = env.state_size
    num_actions = env.action_size

    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    kwargs = {
        "pi_learning_rate": 1e-2,
        "critic_learning_rate": 1e-1,
        "batch_size": 128,
        "entropy_cost": 0.5,
        "max_global_gradient_norm": 20,
    }

    ret = [0]
    max_len = 2000

    with tf.Session() as sess:
        # agents = [DQN(sess, _idx, info_state_size,
        #                   num_actions, hidden_layers_sizes, **kwargs) for _idx in range(2)]
        agents = [PolicyGradient(sess, 0, info_state_size,
                          num_actions, hidden_layers_sizes, **kwargs), MCTSAgent()]
        sess.run(tf.global_variables_initializer())
        print(f"训练回合episode数为:{FLAGS.num_train_episodes}")
        for ep in range(FLAGS.num_train_episodes):
            # 评估模型
            if (ep + 1) % FLAGS.eval_every == 0:
                losses = agents[0].loss
                logging.info("Episodes: {}: Losses: {}, Rewards: {}".format(ep+1, losses, np.mean(ret)))
                # with open('log_pg_{}'.format(os.environ.get('BOARD_SIZE')), 'a+') as log_file:
                with open('log_for_all_models/log_a2c_vs_random/log_pg_{}'.format(os.environ.get('BOARD_SIZE')), 'a+') as log_file:
                    log_file.writelines("{}, {}\n".format(ep+1, np.mean(ret)))
            # 保存模型   这段保存模型的代码得需要自己加上去  不然老是会报错……
            if (ep + 1) % FLAGS.save_every == 0:
                if not os.path.exists(f"{file_path_a2c}"):
                    os.mkdir(f'{file_path_a2c}')
                agents[0].save(checkpoint_root=f'{file_path_a2c}', checkpoint_name='{}'.format(ep+1))
            time_step = env.reset()  # a go.Position object
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id==0:
                    agent_output = agents[player_id].step(time_step)
                else:
                    agent_output = agents[player_id].step(time_step,env)
                action_list = agent_output.action
                time_step = env.step(action_list)
            # print(time_step)
            for agent in agents:
                agent.step(time_step)
            if len(ret) < max_len:
                ret.append(time_step.rewards[0])
            else:
                ret[ep % max_len] = time_step.rewards[0]

        # evaluated the trained agent 评估训练后的智能体
        agents[0].restore(f"{file_path_a2c}/100000")  # 加载训练好的模型

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
        print("time_step的平均值为:",np.mean(ret))

    print('Time elapsed:', time.time()-begin)


if __name__ == '__main__':
    app.run(main)


