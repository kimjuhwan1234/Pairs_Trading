from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os
import warnings
import numpy as np
import pandas as pd
import Metrics as M
import matplotlib.pyplot as plt

# turn off warning
warnings.filterwarnings("ignore")


def momentum_prefix_finder(df: pd.DataFrame):
    possible_prefix = ['', 'Momentum', 'MOM', 'mom', 'Momentum_', 'MOM_', 'mom_', 'Mom_']
    for i in range(1, 10):
        for aposs in possible_prefix:
            if aposs + str(i) in df.columns:
                return aposs
    return ''


def read_and_preprocess_data(input_dir, file) -> pd.DataFrame:
    """
    Only for reading YYYY-MM.csv Files. Recommend using smart_read() for more general cases.
    :param file: YYYY-MM.csv
    :return: DataFrame
    """
    df = pd.read_csv(os.path.join(input_dir, file), index_col=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def generate_PCA_Data(data: pd.DataFrame):
    """
    :param data: momentum_data
    :return: Mom1+PCA_Data
    """
    prefix = momentum_prefix_finder(data)
    mom1 = data.astype(float).loc[:, prefix + '1']

    # mom1을 제외한 mat/PCA(2-49)
    # data = data.drop(columns=[prefix + '1'])
    # data = data.dropna(how='all', axis=1)

    # mom49를 제외한 mat/PCA(1-48)
    data = data.drop(columns=[prefix + '49'])
    data = data.dropna(how='all', axis=1)

    Scaler = StandardScaler()
    data_scaled = Scaler.fit_transform(data)

    pca = PCA(n_components=0.99)
    X_pca = pca.fit_transform(data_scaled)

    columns_pca = [f'PCA_{i}' for i in range(1, X_pca.shape[1] + 1)]
    X_pca_df = pd.DataFrame(X_pca, columns=columns_pca)
    X_pca_df.index = data.index

    df_combined = pd.concat([mom1, X_pca_df], axis=1)
    df_combined.columns = df_combined.columns.astype(str)

    return df_combined


def merge_LS_Table(position, data, LS_merged_df, file):
    data = data[['Firm Name', position]]

    file_column_name = os.path.splitext(file)[0]
    data = data.rename(columns={position: file_column_name})

    if LS_merged_df.empty:
        LS_merged_df = data
    else:
        LS_merged_df = pd.merge(LS_merged_df, data, on='Firm Name', how='outer')

    return LS_merged_df


def product_LS_Table(LS_merged_df: pd.DataFrame, MOM_merged_df: pd.DataFrame,
                     result_df: pd.DataFrame, save: str):
    # Sort LS_Value according to Firm Name
    LS_merged_df.sort_values('Firm Name', inplace=True)

    # Set Firm Name column into index
    LS_merged_df.set_index('Firm Name', inplace=True)
    MOM_merged_df.sort_values('Firm Name', inplace=True)

    # MOM_merged_df=MOM_merged_df.apply(lambda x: np.log(x+1))

    # 마지막 row 버리면 한칸씩 밀어버리는 것과 동치
    LS_merged_df = LS_merged_df.drop(LS_merged_df.columns[-1], axis=1)
    LS_merged_df = LS_merged_df.fillna(0)
    LS_merged_df.columns = MOM_merged_df.columns

    if save == 'LS':
        prod = MOM_merged_df * LS_merged_df
        prod = pd.DataFrame(prod)
        prod = prod.applymap(lambda x: -0.30 if x < -0.30 else x)

        mask = ((LS_merged_df == 1) | (LS_merged_df == -1)) & MOM_merged_df.isna()
        prod[mask] = -0.5

        prod = prod.applymap(lambda x: np.log(x + 1))

        non_zero_count = LS_merged_df.astype(bool).sum()
        column_sums = prod.sum()
        column_means = column_sums.values / non_zero_count.values
        column_means = pd.DataFrame(column_means, index=column_sums.index)
        column_means.fillna(0, inplace=True)
        result_df = pd.concat([result_df, column_means.T], ignore_index=True)

    if save == 'L':
        prod = MOM_merged_df * LS_merged_df[LS_merged_df == 1]
        prod = pd.DataFrame(prod)
        prod = prod.applymap(lambda x: -0.30 if x < -0.30 else x)

        mask = ((LS_merged_df == 1)) & MOM_merged_df.isna()
        prod[mask] = -0.5

        prod = prod.applymap(lambda x: np.log(x + 1))

        non_zero_count = LS_merged_df.astype(bool).sum()
        column_sums = prod.sum()
        column_means = column_sums.values / non_zero_count.values
        column_means = pd.DataFrame(column_means, index=column_sums.index)
        column_means.fillna(0, inplace=True)
        result_df = pd.concat([result_df, column_means.T], ignore_index=True)

    if save == 'S':
        prod = MOM_merged_df * LS_merged_df[LS_merged_df == -1]
        prod = pd.DataFrame(prod)
        prod = prod.applymap(lambda x: -0.30 if x < -0.30 else x)

        mask = ((LS_merged_df == -1)) & MOM_merged_df.isna()
        prod[mask] = -0.5

        prod = prod.applymap(lambda x: np.log(x + 1))

        non_zero_count = LS_merged_df.astype(bool).sum()
        column_sums = prod.sum()
        column_means = column_sums.values / non_zero_count.values
        column_means = pd.DataFrame(column_means, index=column_sums.index)
        column_means.fillna(0, inplace=True)
        result_df = pd.concat([result_df, column_means.T], ignore_index=True)

    return result_df


def plot_UMAP(data, cluster_labels):
    umap_model = UMAP(n_neighbors=10, min_dist=0.5, n_components=2, random_state=42, metric='cosine')
    reduced_embeddings = umap_model.fit_transform(data)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='plasma')
    fig.suptitle(f"UMAP", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.show()


def plot_result(result_df, clustering_name, new_Plot: bool, apply_log: bool):
    result_df = result_df.cumsum(axis=1) if apply_log else result_df.cumprod(axis=1)

    if new_Plot:
        color_dict = {
            'Contrastive_Learning': 'blue',
            'K-mean': 'red',  # Standard red
            'DBSCAN': 'lightcoral',  # Darker shade of red
            'Agglomerative': 'darkred',  # Darkest shade of red

            'Bisecting_K_mean': 'blue',  # Standard blue
            'HDBSCAN': 'steelblue',  # Darker shade of blue
            'BIRCH': 'navy',  # Darkest shade blue
            'ETC': 'navy',

            'OPTICS': 'deepskyblue',  # Bright blue
            'Meanshift': 'royalblue',  # Darker shade of skyblue
            'GMM': 'midnightblue',  # Darkest shade of skyblue

            'Cointegration': 'darkgrey',  # Darker shade of grey
            'Reversal': 'lightgrey',  # Lighter shade of grey
            'FTSE 100': 'grey',  # Standard grey
            'S&P 500': 'grey'
        }

        plt.figure(figsize=(10, 4))
        handles = []
        for key in color_dict:
            if key in result_df.index:
                idx = result_df.index.get_loc(key)
                line, = plt.plot(result_df.columns, result_df.iloc[idx].fillna(method='ffill'),
                                 label=key, color=color_dict[key])
                handles.append(line)

        plt.title(f'{clustering_name} Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Log-returns')

        plt.xticks(rotation=45)
        plt.legend(handles=handles)
        plt.tight_layout()
        plt.show()

    if not new_Plot:

        plt.figure(figsize=(10, 6))
        for i in range(len(result_df)):
            plt.plot(result_df.columns[:], result_df.iloc[i, :].fillna(method='ffill'),
                     label=result_df.iloc[i, 0])

        plt.title(f'{clustering_name} Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Log-returns')

        plt.xticks(rotation=45)
        plt.legend(result_df.index)
        plt.tight_layout()
        plt.show()


def save_and_plot_result(result_df: pd.DataFrame, output_dir, clustering_name, file_names, best: bool):
    if best:
        file = '../Database/ETC/sp500_return_div.csv'
        df = pd.read_csv(file)
        df = df.iloc[0:, :]
        df = df.T
        df.drop('Date', inplace=True)
        df.columns = result_df.columns
        file_names.append('S&P 500')
        result_df = pd.concat([result_df, df], axis=0)

    result_df.index = file_names
    result_df.columns = pd.to_datetime(result_df.columns)
    result_df = result_df.astype(float)
    # result_df = np.log(result_df+1)

    result_modified = pd.DataFrame(
        index=['count', 'annual return mean', 'annual return std', 'cumulative return'],
        columns=result_df.index)

    period = [range(0, 391)]
    Metric = M.metrics(result_df, result_modified, period, total=False)
    Metric.cal_describe()
    Metric.cal_sharpe_ratio(['Shrape Ratio'])
    Metric.cal_t_statistics()
    Metric.cal_down_std(['Down std'])
    Metric.cal_sortino_ratio(['Sortino Ratio'])
    Metric.cal_gross_profit()
    Metric.cal_gross_loss()
    Metric.cal_profit_factor(['Profit Factor'])
    Metric.cal_profitable_years()
    Metric.cal_unprofitable_years()
    Metric.cal_MDD(['Maximum Drawdown'])
    Metric.cal_calmar_ratio(['Calmar Ratio'])
    Metric.cal_monthly_statistics()

    Metric.result_modified.to_csv(os.path.join(output_dir, f'{clustering_name}_statistcs_modified.csv'), index=True)
    result_df.to_csv(os.path.join(output_dir, f'{clustering_name}_result_modified.csv'), index=True)

    plot_result(result_df, clustering_name, best, True)
