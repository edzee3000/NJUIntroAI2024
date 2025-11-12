import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from play import AliensEnvPygame

def extract_features(observation):

    # TODO():提取特征



    grid = observation  #将输入的 observation 赋值给 grid，这是一个二维列表，代表游戏或环境的状态。
    features = []       #一个空列表，用于存储提取的特征。

    def cell_to_feature(cell):#这个内部函数接收一个 cell（网格中的一个单元格），并将其转换为一个特征向量。
        object_mapping = {#object_mapping：一个字典，将网格中的对象（如地板、墙壁、角色等）映射到一个唯一的索引。
            'floor': 0,  #地板
            'wall': 1,   #墙壁
            'avatar': 2,  #角色
            'alien': 3, #外星人
            'bomb': 4,   #炸弹
            'portalSlow': 5,  #慢速传送门
            'portalFast': 6, #快速传送门
            'sam': 7,  #角色
            'base': 8  #基地 
        }
        feature_vector = [0] * len(object_mapping)  #feature_vector：一个列表，初始时所有元素都是0，长度与 object_mapping 中的键值对数量相同。
        for obj in cell:#对于单元格中的每个对象，函数查找其在 object_mapping 中的索引，如果找到，则在 feature_vector 对应的位置设置为1。
            index = object_mapping.get(obj, -1)
            if index >= 0:
                feature_vector[index] = 1
        # print(feature_vector)
        return feature_vector

    def my_cell_to_feature(cell):
        object_mapping = {
            # 'floor': 0,
            'wall': 0,
            'avatar': 1,
            'alien': 2,
            'bomb': 3,
            # 'portalSlow': 5,
            # 'portalFast': 6,
            'sam': 4,
            'base': 5
        }
        object_weight={
            # 'floor': 0.2,
            'wall': 0.5,
            'avatar': 0.8,
            'alien': 1,
            'bomb': 1.2,
            # 'portalSlow': 0.1,
            # 'portalFast': 0.1,
            'sam': 0.2,
            'base': 0.35
        }
        feature_vector = [0] * len(object_mapping)
        for obj in cell:
            index = object_mapping.get(obj, -1)
            if index >= 0:
                feature_vector[index] += object_weight[obj]
        return feature_vector


    for row in grid:#代码通过两层循环遍历 grid 中的每一行和每一列。对于每个 cell，调用 cell_to_feature 函数将其转换为特征向量，并将这个向量扩展到 features 列表中。
        for cell in row:
            # cell_feature = cell_to_feature(cell)
            cell_feature = my_cell_to_feature(cell)
            # print(cell_feature)
            features.extend(cell_feature)
    # 这个函数的作用是将一个二维网格中的每个单元格转换为一个独热编码（one-hot encoding）的特征向量，
    # 然后将所有这些向量合并成一个长的特征向量。这样做的好处是，可以将环境的状态表示为一个固定长度的特征向量，
    # 便于后续的机器学习模型处理。例如，在强化学习中，这个特征向量可以作为输入提供给神经网络来学习策略。
    return np.array(features)


# #因为是onehot独热编码因此不适合直接用PCA降维
def SparsePCA_Kmeans(X):
    # 使用稀疏PCA进行降维
    sparse_pca = SparsePCA(n_components=None, random_state=0)
    X_sparse_pca = sparse_pca.fit_transform(X)
    return X_sparse_pca


def main():
    data_list = [
        # 'game_records_lvl0_2024-xx-xx_xx-xx-xx', # 修改路径为你的数据
        # 'game_records_lvl0_2024-yy-yy_yy-yy-yy',
        'game_records_lvl0_2024-10-25_13-04-42',
        'game_records_lvl0_2024-10-25_13-41-11',
        'game_records_lvl0_2024-10-29_18-37-56',
        'game_records_lvl0_2024-10-29_18-38-03',
        'game_records_lvl0_2024-10-29_18-38-38',
        'game_records_lvl0_2024-10-29_18-39-38',
        'game_records_lvl0_2024-10-29_18-43-03',
        'game_records_lvl0_2024-10-29_18-45-22',
        'game_records_lvl0_2024-10-29_18-45-32',
        'game_records_lvl0_2024-10-29_18-47-38',
        'game_records_lvl0_2024-10-29_18-52-10',
    ]
    data = []
    for data_load in data_list:
        with open(os.path.join('logs','PlayDatas', data_load, 'data.pkl'), 'rb') as f:
            data += pickle.load(f)

    X = []
    y = []
    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)




    # 可不可以在这里加一个将X进行缩小？？？因为有些位置永远为0有些位置永远为1


    # X=SparsePCA_Kmeans(X) 本来想先PCA降维的  但是工作量好像有点大
    
    clf = RandomForestClassifier(n_estimators=100)
    # clf = RandomForestClassifier(n_estimators=150,max_depth=15,max_features=15)
    # clf = DecisionTreeClassifier(max_depth=15,max_features=15)
    # clf=KNeighborsClassifier(n_neighbors=15)
    clf.fit(X, y)

    # 预测测试集
    y_pred = clf.predict(X)

    # 计算准确率
    # accuracy = accuracy_score(y, y_pred)
    # print(f"KNN 分类器准确率: {accuracy}")




    # env = AliensEnvPygame(level=0, render=False,flag_not_AI=True)
    env = AliensEnvPygame(level=0, render=False)

    name_module='Modules'
    path_module = f'logs/{name_module}/game_records_lvl{env.level}_{env.timing}'
    os.makedirs(path_module, exist_ok=True)    #重新创建新的模型的路径
    with open(f'{path_module}/gameplay_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # with open(f'{env.log_folder}/gameplay_model.pkl', 'wb') as f:
    #     pickle.dump(clf, f)


    print("模型训练完成")

if __name__ == '__main__':
    main()
