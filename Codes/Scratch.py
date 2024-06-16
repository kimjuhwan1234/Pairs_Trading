from glob import glob
from PCA_and_ETC import *
from tqdm.auto import tqdm
from scipy.stats.mstats import winsorize

import copy
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def investment_strategy(return_data_frame, number_of_clusters, threshold, batch=64, bins=32, hidden=128):
    cluster_dir = f'../Database/Clustering_Result/Contrastive_Learning/{number_of_clusters}_characteristics_us_batch_{batch}_bins_{bins}_hidden_{hidden}'
    cluster_results = glob(cluster_dir + '/*')
    cluster_results.sort()

    all_months = return_data_frame.columns

    cumulative_return = 0
    cumulative_returns_list = []
    log_returns_list = []

    for idx, current_month in enumerate(tqdm(all_months)):

        assert str(cluster_results[idx][-11:].split('.')[0]) == str(current_month), '월이 일치하지 않음'

        # 다음 달
        next_month = all_months[idx + 1] if idx + 1 < len(all_months) else None
        if next_month is None:
            continue

        cluster_data = pd.read_csv(cluster_results[idx], index_col=0)
        cluster_data.sort_values(by='mom1', ascending=False, inplace=True)

        clusters = cluster_data.sort_values(by=['clusters', 'mom1'], ascending=[True, False])
        spread_vec = (clusters.reset_index()['mom1'] -
                      clusters.sort_values(by=['clusters', 'mom1'],
                                           ascending=[True, True]).reset_index()['mom1'])

        spreadStd = spread_vec.std()

        equityReturnsCopy = copy.deepcopy(return_data_frame)

        ls_result = []

        # 스프레드 계산
        for _, group in cluster_data.groupby('clusters'):
            returnsAscend = group.sort_values(by='mom1', ascending=True)['mom1']
            returnsDescend = group.sort_values(by='mom1', ascending=False)['mom1']
            spreadArray = returnsDescend.values - returnsAscend.values

            group['spread'] = spreadArray

            inPortfolio = group['spread'].abs() > spreadStd * threshold
            ls_result.append(list(group[inPortfolio].index))
            Long_or_Short = (-group['spread'] / (group['spread'].abs()))

            long_firms = group[inPortfolio & (Long_or_Short == 1)]
            short_firms = group[inPortfolio & (Long_or_Short == -1)]
            cash_firms = group[~inPortfolio]

            equityReturnsCopy.loc[short_firms.index, next_month] *= -1
            equityReturnsCopy.loc[cash_firms.index, next_month] *= 0
            is_nan = equityReturnsCopy.loc[:, current_month].isna()
            equityReturnsCopy.loc[is_nan, next_month] = np.nan

        ls_result = [p for sublist in ls_result for p in sublist]
        num_invest = len(ls_result)

        # 포트폴리오 가치 계산
        earning_with_cash = equityReturnsCopy.loc[:, next_month]
        earning = earning_with_cash[earning_with_cash != 0].dropna().sum() / num_invest

        log_return = np.log(1 + earning)
        log_returns_list.append(log_return)
        cumulative_return = np.exp(np.log(1 + cumulative_return) + log_return) - 1
        cumulative_returns_list.append(cumulative_return)

    print("포트폴리오 수익률: ", cumulative_return)
    return cumulative_return, cumulative_returns_list, log_returns_list


# cumulative_return_10, cumulative_returns_list_10, log_returns_list_10 = investment_strategy(df_winsorized, number_of_clusters=10, threshold=2)
# cumulative_return_20, cumulative_returns_list_20, log_returns_list_20 = investment_strategy(df_winsorized_final, number_of_clusters=20, threshold=2)
# cumulative_return_30, cumulative_returns_list_30, log_returns_list_30 = investment_strategy(df_winsorized_final, number_of_clusters=30, threshold=2)
# cumulative_return_50, cumulative_returns_list_50, log_returns_list_50 = investment_strategy(df_winsorized_final, number_of_clusters=50, threshold=2)
# cumulative_return_100, cumulative_returns_list_100, log_returns_list_100 = investment_strategy(df_winsorized_final, number_of_clusters=100, threshold=2)


if __name__ == '__main__':
    df_winsorized = pd.read_csv('../Database/winsorized_mom1_data_combined_adj_close.csv', index_col='Firm Name')
    cumulative_return_1, cumulative_returns_list_1, log_returns_list_1 = investment_strategy(
        df_winsorized, number_of_clusters=10, threshold=2, batch=64, bins=32, hidden=128)

    print()

    files = sorted(filename for filename in os.listdir(f'../Database/characteristics_US') if filename.endswith('.csv'))

    a = 0
    for file in files:
        print(file)
        data = read_and_preprocess_data(f'../Database/characteristics_US', file)


    data=pd.read_csv('../Database/data2.csv')
    data[data['gvkey']==50906]


    min_max = False
    if min_max:
        df = pd.read_csv('../Files/mom1_data_combined_adj_close2.csv')
        df = df.iloc[:, 1:]

        result = pd.DataFrame(df.max())
        result.to_csv('../Files/max_mom1.csv')

        result = pd.DataFrame(df.min())
        result.to_csv('../Files/min_mom1.csv')
