#导入必要的模块：random用于生成随机数，time用于计算时间，Go是从environment包导入的围棋环境类，RandomAgent是从agent包导入的随机智能体类。
import random
from environment.GoEnv import Go
import time
from agent.agent import RandomAgent


if __name__ == '__main__':
    begin = time.time()  #记录开始时间。
    env = Go()   #创建一个围棋环境实例。
    agents = [RandomAgent(idx) for idx in range(2)]  #创建一个包含两个随机智能体（RandomAgent）的列表，分别对应两个玩家（例如，玩家0和玩家1）。

    for ep in range(10):  #运行10次游戏
        time_step = env.reset()  #重置环境，重新开始一次新的游戏，并获得初始的时间步。
        while not time_step.last():  #如果游戏没有结束重复进行
            player_id = time_step.observations["current_player"]#获取当前玩家的ID
            if player_id == 0: #根据当前玩家的ID，让相应的智能体采取行动。
                agent_output = agents[player_id].step(time_step)
            else:
                agent_output = agents[player_id].step(time_step)
            action_list = agent_output.action   #从智能体的输出中获取动作列表。
            time_step = env.step(action_list)  #使用这个动作列表在环境中执行一步。
            print(time_step.observations["info_state"][0])  #打印当前信息状态（可能是棋盘状态）。

        # Episode is over, step all agents with final info state.
        for agent in agents:#在游戏回合结束时，通知所有智能体，这样它们可以处理最终的状态（例如，更新策略或学习）
            agent.step(time_step)
        print(f"第{ep+1}次游戏结束")
    print('Time elapsed:', time.time()-begin)
