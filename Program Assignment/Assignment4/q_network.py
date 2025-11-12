import torch
import torch.nn as nn  #提供神经网络相关的模块和功
import torch.nn.functional as F  #提供神经网络中常用的函数，如激活函数、损失函数等
#定义了一个深度神经网络模型，用于强化学习中的Q网络（Q-Network）
# Define the neural network model
class QNetwork(nn.Module):  #QNetwork类继承自nn.Module，这是PyTorch中所有神经网络模块的基类。
    def __init__(self, state_dim, action_dim, seed, fc_units=128, device="cpu"):
        # state_dim：输入状态的维度。
        # action_dim：输出动作的维度。
        # seed：用于初始化随机数生成器的种子，确保模型初始化的可重复性。
        # fc_units：全连接层中神经元的数量，默认为128。
        # device：指定模型应该使用的设备（CPU或GPU），默认为"cpu"。
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)#设置随机种子以确保模型初始化的可重复性。
        self.fc1 = nn.Linear(state_dim, fc_units)#第一个全连接层，输入维度为state_dim，输出维度为fc_units。
        self.fc2 = nn.Linear(fc_units, fc_units)#第二个全连接层，输入维度为fc_units，输出维度也为fc_units。
        self.fc3 = nn.Linear(fc_units, action_dim)#第三个全连接层，输入维度为fc_units，输出维度为action_dim。
        self.to(device)#使用.to(device)方法将模型移动到指定的设备上。

    def forward(self, state): #接收一个参数state，表示输入的状态
        x = F.relu(self.fc1(state))#将输入状态传递给第一个全连接层self.fc1，并应用ReLU激活函数。
        x = F.relu(self.fc2(x))#将激活后的输出传递给第二个全连接层self.fc2，并再次应用ReLU激活函数。
        return self.fc3(x)#第二个全连接层的输出传递给第三个全连接层self.fc3，得到最终的输出。
        #返回第三个全连接层的输出，这代表了在给定状态下每个动作的Q值。