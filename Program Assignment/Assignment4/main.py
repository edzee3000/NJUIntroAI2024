import gym  #用于创建和运行强化学习环境
from collections import deque  #用于实现经验回放缓冲区
import random  #用于生成随机数。
import math
import argparse  #用于解析命令行参数。
import torch  #实现深度学习模型。

# import matplotlib
# matplotlib.use('TkAgg')  # 或者其他支持 GUI 的后端
import matplotlib.pyplot as plt
import numpy as np
from agent import DQNAgent, DDQNAgent   #agent模块包含了DQNAgent和DDQNAgent两个类，用于实现DQN和DDQN算法。

#基于DQN（Deep Q-Network）和DDQN（Double DQN）算法的智能体来训练和评估在Gym环境中的CartPole-v1任务
Episode=[]
Returns=[]
TrainingLoss=[]

def parser(): #定义命令行参数解析函数parser()
    #该函数使用argparse库来定义和解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", type=str, default="ddqn")  #智能体类型  默认使用DQN
    # parser.add_argument("--num_episodes", type=int, default=600) #训练的回合数
    parser.add_argument("--num_episodes", type=int, default=600) #训练的回合数
    parser.add_argument("--max_steps_per_episode", type=int, default=500) #每回合的最大步数
    # parser.add_argument("--epsilon_start", type=float, default=0.9)  #探索起始值
    parser.add_argument("--epsilon_start", type=float, default=0.4)
    # parser.add_argument("--epsilon_end", type=float, default=0.5)  #探索终止值
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    # parser.add_argument("--epsilon_decay", type=float, default=1000)#探索衰减率
    parser.add_argument("--epsilon_decay", type=float, default=5000)
    # parser.add_argument("--gamma", type=float, default=0.99)  #折扣因子
    parser.add_argument("--gamma", type=float, default=0.99)
    # parser.add_argument("--lr", type=float, default=1e-1) #learning rate  学习率
    parser.add_argument("--lr", type=float, default=1e-2)
    # parser.add_argument("--buffer_size", type=int, default=10000)#经验回放缓冲区大小
    parser.add_argument("--buffer_size", type=int, default=20000)
    # parser.add_argument("--batch_size", type=int, default=32)#批量大小
    parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--update_frequency", type=int, default=10)#学习频率
    parser.add_argument("--update_frequency", type=int, default=20)
    
    args = parser.parse_args()
    return args


def eval_policy(agent):#定义策略评估函数eval_policy(agent)
    #该函数用于评估智能体的策略，通过运行一个回合来计算总回报。
    state = env.reset()
    done = False
    return_ = 0
    while not done:
        action = agent.act(state, eps=0.)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        return_ += reward
    # print(f"Return {return_}") 
    return return_


def train(args, agent):#定义训练函数train(args, agent)  主要的训练循环
    #包括初始化环境、经验回放缓冲区，以及根据参数选择DQN或DDQN智能体。
    # Training loop
    step_count = 0  #step_count: 用于记录训练过程中的总步数，用于计算探索率（epsilon）的衰减。
    for episode in range(args.num_episodes):
        #在每个回合中，智能体执行动作，并将经验存储在缓冲区中。当缓冲区中的经验足够时，从中随机抽取一个批次进行学习。
        # Reset the environment 重置环境以获得初始状态
        state = env.reset()
        # Run one episode 运行一个回合
        losses = []#初始化回合内的损失列表losses和回报return
        return_ = 0

        # env.render()##################################
        for step in range(args.max_steps_per_episode):
            # Choose and perform an action
            step_count += 1#更新step_count。
            epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * math.exp(-1. * step_count / args.epsilon_decay)#计算当前的探索率epsilon，它随着step_count的增加而衰减
            action = agent.act(state, epsilon)#使用智能体的act方法选择一个动作，该方法考虑了当前的探索率
            # action = agent.act_no_explore(state, epsilon)
            next_state, reward, done, _ = env.step(action)#执行选定的动作，并观察环境返回的下一个状态、奖励、是否完成以及额外信息
            
            buffer.append((state, action, reward, next_state, done))#将经验（状态、动作、奖励、下一个状态、是否完成）添加到经验回放缓冲区buffer中。
            
            # env.render()##################################
            if len(buffer) >= args.batch_size:#如果缓冲区中的经验数量足够进行一次批量学习，
                batch = random.sample(buffer, args.batch_size)#，则从缓冲区中随机抽取一个批次
                # Update the agent's knowledge更新当前智能体的知识
                loss = agent.learn(batch, args.gamma)#并使用智能体的learn方法进行学习。
                losses.append(loss)
            return_ += reward#累加回报。
            
            state = next_state#更新状态为下一个状态。
            
            # Check if the episode has ended
            if done:#如果环境指示回合结束（done为True），则跳出循环。
                break
        #每个回合结束后，计算并打印训练损失和评估回报。
        loss = torch.mean(torch.tensor(losses))#计算回合内的平均损失，并将损失列表losses转换为PyTorch张量，然后计算其平均值。
        eval_return = eval_policy(agent)#使用eval_policy函数评估当前智能体的策略，计算无探索（epsilon=0）情况下的回报。

        print(f"Episode {episode + 1} Step {step + 1}: Training Loss {loss}, Return {eval_return}")
        global Episode,Returns,TrainingLoss
        Episode.append(episode + 1)
        Returns.append(eval_return)
        TrainingLoss.append(loss)

def draw():
    global Episode,Returns,TrainingLoss
    Episode=np.array(Episode)
    Returns=np.array(Returns)
    TrainingLoss=np.array(TrainingLoss)
    lr=0.99
    gamma=0.99
    start=0.4
    end=0.1
    decay=5000
    batch=64
    plt.plot(Episode,Returns)
    plt.suptitle("DDQN Returns")
    plt.title(f"lr={lr},gamma={gamma},start={start},end={end},decay={decay},batch={batch}")
    plt.savefig('./Report/pictures/FigureReturn.png')

    plt.figure()
    plt.plot(Episode,TrainingLoss)
    plt.suptitle("DDQN TrainingLoss")
    plt.title(f"lr={lr},gamma={gamma},start={start},end={end},decay={decay},batch={batch}")
    plt.savefig('./Report/pictures/FigureLoss.png')
    # plt.show()




if __name__ == "__main__":
    args = parser()#解析命令行参数
    # args = parser("dqn",3000)
    # Set up the environment
    env = gym.make("CartPole-v1")#设置环境为CartPole-v1   用于解决CartPole-v1任务

    buffer = deque(maxlen=args.buffer_size)#初始化经验回放缓冲区

    # Initialize the DQNAgent 根据参数选择并初始化DQN或DDQN智能体。
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    if args.agent_name == "dqn":
        agent = DQNAgent(input_dim, output_dim, seed=1234, lr = args.lr)
    elif args.agent_name == "ddqn":
        agent = DDQNAgent(input_dim, output_dim, seed=1234, lr = args.lr)
    else:
        assert False, "Not Implement agent!"
    train(args, agent)#调用train函数开始训练过程
    env.close()
    draw()