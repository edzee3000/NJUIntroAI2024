import random

import numpy as np
import torch
from collections import namedtuple, deque  #分别用于创建简单的轻量级对象和用于创建一个双端队列存储经验数据
#定义了一个名为ReplayBuffer的类，用于实现经验回放（Experience Replay）机制，这在强化学习中非常常见，尤其是在深度Q网络（DQN）算法中
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, batch_size, buffer_size, seed, device="cpu"):
        # state_dim：状态的维度。
        # action_dim：动作的维度。
        # batch_size：每次从回放缓冲区中采样的批次大小。
        # buffer_size：回放缓冲区的最大容量。
        # seed：用于初始化随机数生成器的种子。
        # device：指定用于存储和计算数据的设备（CPU或GPU），默认为"cpu"。
        self.action_size = state_dim #这里是不是有问题？？？应该是self.state_size = state_dim？？？？
        self.memory = deque(maxlen=buffer_size) #使用deque创建一个经验回放缓冲区，最大长度为buffer_size。
        self.batch_size = batch_size#存储采样的批次大小。
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])#使用namedtuple定义一个经验元组，包含状态、动作、奖励、下一个状态和是否完成。
        self.seed = random.seed(seed)

    #用于将新的经验数据添加到回放缓冲区中。  
    def add(self, state, action, reward, next_state, done):#接收状态、动作、奖励、下一个状态和是否完成作为参数。
        e = self.experience(state, action, reward, next_state, done)#创建一个experience对象，并将其添加到memory队列的末尾
        self.memory.append(e)

    def sample(self):#从回放缓冲区中随机采样一个批次的经验数据。
        experiences = random.sample(self.memory, k=self.batch_size) #使用random.sample从memory中随机选择batch_size个经验。
        #将每个经验中的状态、动作、奖励、下一个状态和是否完成分别堆叠成NumPy数组，然后转换为PyTorch张量，并移动到指定的设备上
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        #返回一个包含状态、动作、奖励、下一个状态和是否完成的元组。
        return (states, actions, rewards, next_states, dones)

    def __len__(self):#返回回放缓冲区中当前存储的经验数量。
        return len(self.memory)
    #这个ReplayBuffer类是DQN算法中的一个关键组件，它允许智能体从其经验中随机采样数据来打破数据之间的相关性，从而提高学习的稳定性和效率。