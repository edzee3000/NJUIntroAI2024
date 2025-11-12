import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from q_network import QNetwork
from replay_buffer import ReplayBuffer

# Define the DQN agent class
class DQNAgent:
    # Initialize the DQN agent
    def __init__(self, state_dim, action_dim, seed, lr, device="cpu"):
        #接收状态维度、动作维度、随机种子、学习率以及设备（默认为CPU）作为参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(seed)
        self.device = device
        #初始化本地Q网络和目标Q网络，它们都是通过QNetwork类创建的，并将它们移动到指定的设备上
        self.qnetwork_local = QNetwork(state_dim, action_dim, seed).to(device)
        self.qnetwork_target = QNetwork(state_dim, action_dim, seed).to(device)
        #初始化本地Q网络和目标Q网络，它们都是通过QNetwork类创建的，并将它们移动到指定的设备上
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)
        #初始化一个经验回放缓冲区ReplayBuffer，用于存储经验数据。
        self.memory = ReplayBuffer(state_dim, action_dim, buffer_size=int(1e5), batch_size=64, seed=seed)
        self.t_step = 0#设置一个时间步计数器t_step。

    def step(self, state, action, reward, next_state, done):
        #接收当前状态、执行的动作、获得的奖励、下一个状态以及是否完成的状态作为参数。
        self.memory.add(state, action, reward, next_state, done)#将经验添加到经验回放缓冲区中。
        #每隔一定的步数（这里设置为4步），从经验回放缓冲区中抽取一个批次的经验进行学习。
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > 64:
                experiences = self.memory.sample()
                self.learn(experiences, gamma=0.99)

    # Choose an action based on the current state
    def act(self, state, eps=0.):# #接收当前状态和探索率eps作为参数。
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        #将状态转换为张量，并在评估模式下运行本地Q网络以获取动作值。
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        #根据探索率决定是选择具有最大动作值的动作还是随机选择一个动作。
        if np.random.random() > eps:
            return action_values.argmax(dim=1).item()
        else:
            return np.random.randint(self.action_dim)
    #与act方法类似，但不进行探索，总是选择具有最大动作值的动作。
    def act_no_explore(self, state, eps=0.):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        return action_values.argmax(dim=1).item()

    # Learn from batch of experiences
    def learn(self, experiences, gamma):#接收经验批次和折扣因子gamma作为参数。
        device = self.device
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        #将经验数据转换为张量，并计算目标Q值和预期的Q值。
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)
        #使用均方误差损失函数计算损失，并通过反向传播更新本地Q网络的参数。
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #调用soft_update方法来软更新目标Q网络的参数。
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)
        return loss

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, seed, lr, device="cpu"):
        super().__init__(state_dim, action_dim, seed, lr, device)
    #继承自DQNAgent类，并调用其初始化方法。

    # Learn from batch of experiences  重写DQNAgent类的learn方法。
    def learn(self, experiences, gamma):
        #在计算目标Q值时采用了双网络结构，以提高学习的稳定性和性能
        device = self.device
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        #在计算目标Q值时，使用本地Q网络来选择在下一个状态上具有最大动作值的动作，然后使用目标Q网络来获取该动作的Q值。
        # 这是DDQN算法的关键区别，它有助于减少Q值的高估。
        action_max = torch.argmax(self.qnetwork_local(next_states), dim=1).unsqueeze(1)
        Q_targets = rewards + (gamma * torch.gather(self.qnetwork_target(next_states).detach(), 1, action_max) * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)
        return loss
#这两个类都依赖于QNetwork类来定义Q网络的结构，以及ReplayBuffer类来存储和采样经验数据。


