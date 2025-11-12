import os
import sys
import pygame
import pickle
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from play import AliensEnvPygame
from learn import extract_features

def main():

    # clf =  KNeighborsClassifier(n_neighbors=15)

    # clf = RandomForestClassifier(n_estimators=150,max_depth=15,max_features=15)
    clf = RandomForestClassifier(n_estimators=100)
    # clf = RandomForestClassifier(n_estimators=150)
    # clf = DecisionTreeClassifier(max_depth=15,max_features=15)
    pygame.init()

    # env = AliensEnvPygame(level=0, render=False)  #我们让随机森林分类器作为level=0
    env = AliensEnvPygame(level=1, render=False)     #让决策树分类器作为level=1

    # 加载模型
    # model_path = 'logs\game_records_lvl0_2024-xx-xx_xx-xx-xx\gameplay_model.pkl' # 替换为你的模型的路径
    # model_path = f'logs/Modules/game_records_lvl0_2024-10-25_13-24-30/gameplay_model.pkl'
    # model_path = f'logs/Modules/game_records_lvl0_2024-10-29_18-40-55/gameplay_model.pkl'
    # model_path =f'logs/Modules/game_records_lvl0_2024-10-29_19-38-35/gameplay_model.pkl'
    # model_path =f'logs/Modules/game_records_lvl0_2024-10-29_18-51-01/gameplay_model.pkl'
    # model_path =f'logs/Modules/game_records_lvl0_2024-10-29_19-44-00/gameplay_model.pkl'
    # model_path =f'logs/Modules/game_records_lvl0_2024-10-31_20-13-27/gameplay_model.pkl'
    # model_path =f'logs/Modules/game_records_lvl0_2024-11-05_13-26-19/gameplay_model.pkl'
    model_path =f'logs/Modules/game_records_lvl0_2024-11-05_17-05-20/gameplay_model.pkl'


    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    print("模型加载完成")

    observation = env.reset()

    grid_image = env.do_render()

    mode = grid_image.mode
    size = grid_image.size
    data_image = grid_image.tobytes()
    pygame_image = pygame.image.fromstring(data_image, size, mode)

    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Aliens Game - AI Playing')

    screen.blit(pygame_image, (0, 0))
    pygame.display.flip()

    done = False
    total_score = 0
    step = 0
    while not done:
        features = extract_features(observation)
        features = features.reshape(1, -1)

        action = clf.predict(features)[0]

        observation, reward, game_over, info = env.step(action)
        total_score += reward
        print(f"Step: {step}, Action taken: {action}, Reward: {reward}, Done: {game_over}, Info: {info}")
        step += 1

        grid_image = env.do_render()
        mode = grid_image.mode
        size = grid_image.size
        data_image = grid_image.tobytes()
        pygame_image = pygame.image.fromstring(data_image, size, mode)

        screen.blit(pygame_image, (0, 0))
        pygame.display.flip()

        if game_over or step > 500:
            print("游戏结束!")
            print(f"信息: {info}，分数：{total_score}")
            done = True

        pygame.time.delay(100)

    #保存最终测试结果    
    name_testresult='TestResult'
    path_result=f'logs/{name_testresult}/game_records_lvl{env.level}_{env.timing}'
    os.makedirs(path_result, exist_ok=True)    #重新创建保存游戏结果的路径
    env.save_gif(filename=f'replay_ai.gif',path=path_result)


    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
