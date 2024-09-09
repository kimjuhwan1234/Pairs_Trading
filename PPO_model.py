import gym
import numpy as np
from gym import spaces
from collections import deque

import os
import numpy as np
import pandas as pd
from utils.PCA_and_ETC import read_and_preprocess_data, momentum_prefix_finder

input_dir = 'Database/Clustering_Result/K_mean'
subdirectories = [d for d in os.listdir(input_dir)]
MOM_merged_df = pd.read_csv('Database/Momentum1_Winsorized.csv', index_col=0)
MOM_merged_df.drop(MOM_merged_df.columns[0], axis=1, inplace=True)


class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.return_que = deque()
        self.data = []
        self.next_file = []

    def step(self, action, data, next_file, initial=None):
        prefix = momentum_prefix_finder(data)

        mom1_col_name = prefix + '1'

        cluster = []
        for j in range(1 + max(set(data['clusters']))):
            indices = list(data[data['clusters'] == j].index)
            cluster.append(indices)

        clusters = []
        for i in range(len(cluster)):
            for firms in cluster[i]:
                clusters.append([firms, i])

        clusters = pd.DataFrame(clusters, columns=['Firm Name', 'Cluster Index'])
        clusters = clusters.set_index('Firm Name')
        clusters['Momentum_1'] = data[mom1_col_name]
        # clusters = clusters.sort_values(by=['Cluster Index', 'Momentum_1'], ascending=[True, False])
        clusters = clusters.sort_values(by=['Cluster Index', 'Momentum_1', 'Firm Name'], ascending=[True, False, True])

        spread_vec = (clusters.reset_index()['Momentum_1'] -
                      clusters.sort_values(by=['Cluster Index', 'Momentum_1', 'Firm Name'],
                                           ascending=[True, True, True]).reset_index()['Momentum_1'])

        clusters = clusters.reset_index()
        clusters['spread'] = spread_vec
        # clusters['spread'][clusters['Cluster Index'] != 0].std()
        clusters['in_portfolio'] = (clusters['spread'].abs() > clusters['spread'].std() * action)
        clusters['Long Short'] = clusters['in_portfolio'] * (-clusters['spread'] / clusters['spread'].abs())
        clusters['Long Short'] = clusters['Long Short'].fillna(0)

        # clusters = clusters.drop(columns=['spread', 'in_portfolio'])
        clusters.loc[clusters['Cluster Index'] == 0, 'Long Short'] = 0
        clusters['Long'] = clusters['Long Short'].apply(lambda x: 1 if x == 1 else 0)
        clusters['Short'] = clusters['Long Short'].apply(lambda x: -1 if x == -1 else 0)

        clusters.set_index('Firm Name', inplace=True)

        MOM_merged_df.sort_values('Firm Name', inplace=True)

        # reward = (MOM_merged_df.loc[clusters.index, next_file[:-4]] * clusters['Long Short']).sum()
        return_df=MOM_merged_df.loc[:, next_file[:-4]]
        prod = MOM_merged_df.loc[clusters[clusters['Long Short'] != 0].index, next_file[:-4]]
        prod = prod * clusters['Long Short']
        prod = prod.apply(lambda x: -0.30 if x < -0.30 else x)

        mask = ((clusters['Long Short'] == 1) | (clusters['Long Short'] == -1)) & prod.isna()
        prod[mask] = -0.5

        prod = prod.apply(lambda x: np.log(x + 1))

        non_zero_count = clusters['Long Short'].astype(bool).sum()
        column_sums = prod.sum()
        reward = column_sums / non_zero_count

        if (not initial) & (len(self.return_que) >= 12):
            # self.return_que.popleft()
            self.return_que.append(reward)
            d_replaced = [0 if np.isnan(x) else x for x in self.return_que]
            sf = (np.exp(np.mean(d_replaced) * 12) - 1) / (np.exp(np.std(d_replaced) * np.sqrt(12)) - 1)
            self.state = np.array(
                [np.mean(return_df.fillna(0)), np.std(return_df.fillna(0)), clusters['spread'].mean(), clusters['spread'].std()])
            reward = (np.exp(np.mean(d_replaced) * 12) - 1)
            return self.state, reward, False, {}


        elif initial:
            d_replaced = [0 if np.isnan(x) else x for x in self.return_que]
            sf = (np.exp(np.mean(d_replaced) * 12) - 1) / (np.exp(np.std(d_replaced) * np.sqrt(12)) - 1)
            self.state = np.array(
                [np.mean(return_df.fillna(0)), np.std(return_df.fillna(0)), clusters['spread'].mean(), clusters['spread'].std()])
            return self.state

        else:
            return reward, False, {}

    def reset(self, data, next_file, initial=None):
        if not initial:
            r, done, _ = self.step(1, data, next_file)
            self.return_que.append(r)
        elif initial:
            initial_state = self.step(1, data, next_file, True)
            return initial_state

    def past_return(self):
            return [0 if np.isnan(x) else x for x in self.return_que]
