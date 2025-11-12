from absl import logging, flags, app
from environment.GoEnv import Go
import time, os
import numpy as np
from algorimths.policy_gradient import PolicyGradient
import tensorflow as tf

FLAGS = flags.FLAGS

# flags.DEFINE_integer("num_train_episodes", 100000,
#                      "Number of training episodes for each base policy.")
flags.DEFINE_integer("num_train_episodes", 100000,
                     "Number of training episodes for each base policy.")
flags.DEFINE_integer("num_eval", 1000,
                     "Number of evaluation episodes")
flags.DEFINE_integer("eval_every", 2000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer("learn_every", 128,
                     "Episode frequency at which the agents learn.")
flags.DEFINE_integer("save_every", 20000,
                     "Episode frequency at which the agents save the policies.")
#保存模型频率  靠这一句也得后来自己去加上……无语死了……   而且由于训练episodes数量逐渐增大，保存频率也需要减缓……  没事也就多加个0的事儿嘛
flags.DEFINE_list("hidden_layers_sizes", [
    128, 256
], "Number of hidden units in the policy-net and critic-net.")



file_path_a2c="saved_model/a2c_vs_random_model"
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
    import agent.agent as agent
    ret = [0]
    max_len = 2000

    with tf.Session() as sess:
        # agents = [DQN(sess, _idx, info_state_size,
        #                   num_actions, hidden_layers_sizes, **kwargs) for _idx in range(2)]
        agents = [PolicyGradient(sess, 0, info_state_size,
                          num_actions, hidden_layers_sizes, **kwargs), agent.RandomAgent(1)]
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
                agent_output = agents[player_id].step(time_step)
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
