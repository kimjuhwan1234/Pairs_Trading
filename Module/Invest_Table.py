from utils.PCA_and_ETC import *


class Invest_Table:

    def __init__(self, data: pd.DataFrame):
        self.PCA_Data = data
        self.prefix = momentum_prefix_finder(self.PCA_Data)
        self.table = []

    def ls_table(self, cluster: list, output_dir, file, save=True, raw=False):
        if raw:
            mom1_col_name = self.prefix + '1'
        else:
            mom1_col_name = '0'

        clusters = []
        for i in range(len(cluster)):
            for firms in cluster[i]:
                clusters.append([firms, i])

        clusters = pd.DataFrame(clusters, columns=['Firm Name', 'Cluster Index'])

        clusters = clusters.set_index('Firm Name')

        # series_no_nan = self.PCA_Data[mom1_col_name].dropna()
        # winsorized_values = winsorize(series_no_nan, limits=[0.05, 0.05])
        # self.PCA_Data[mom1_col_name][series_no_nan.index] = winsorized_values
        clusters['Momentum_1'] = self.PCA_Data[mom1_col_name]

        # clusters = clusters.sort_values(by=['Cluster Index', 'Momentum_1'], ascending=[True, False])
        clusters = clusters.sort_values(by=['Cluster Index', 'Momentum_1', 'Firm Name'], ascending=[True, False, True])

        spread_vec = (clusters.reset_index()['Momentum_1'] -
                      clusters.sort_values(by=['Cluster Index', 'Momentum_1', 'Firm Name'],
                                           ascending=[True, True, True]).reset_index()['Momentum_1'])

        clusters = clusters.reset_index()
        clusters['spread'] = spread_vec
        # clusters['spread'][clusters['Cluster Index'] != 0].std()
        clusters['in_portfolio'] = (clusters['spread'].abs() > clusters['spread'].std())
        clusters['Long Short'] = clusters['in_portfolio'] * (-clusters['spread'] / clusters['spread'].abs())
        clusters['Long Short'] = clusters['Long Short'].fillna(0)

        clusters = clusters.drop(columns=['spread', 'in_portfolio'])
        clusters.loc[clusters['Cluster Index'] == 0, 'Long Short'] = 0
        clusters['Long'] = clusters['Long Short'].apply(lambda x: 1 if x == 1 else 0)
        clusters['Short'] = clusters['Long Short'].apply(lambda x: -1 if x == -1 else 0)
        clusters.sort_values('Cluster Index', inplace=True)
        clusters = clusters[['Firm Name', 'Momentum_1', 'Long Short', 'Long', 'Short', 'Cluster Index']]

        if save:
            clusters.to_csv(os.path.join(output_dir, file), index=False)
            print(f'Exported to {output_dir}!')

        self.table = clusters

    def reversal_table(self, data: pd.DataFrame, output_dir, file, save=True):
        """

        :param data:
        :param output_dir:
        :param file:
        :param save:
        :return:
        """
        LS_table_reversal = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short'])
        firm_lists = data.index
        firm_sorted = sorted(firm_lists, key=lambda x: data.loc[x, self.prefix + '1'])
        long_short = [0] * len(firm_sorted)
        t = int(len(firm_lists) * 0.1)
        for i in range(t):
            long_short[i] = 1
            long_short[-i - 1] = -1

        for i, firm in enumerate(firm_sorted):
            LS_table_reversal.loc[len(LS_table_reversal)] = [firm, data.loc[firm, self.prefix + '1'], long_short[i]]

        LS_table_reversal['Long'] = LS_table_reversal['Long Short'].apply(lambda x: 1 if x == 1 else 0)
        LS_table_reversal['Short'] = LS_table_reversal['Long Short'].apply(lambda x: -1 if x == -1 else 0)

        if save:
            # Save the output to a CSV file in the output directory
            LS_table_reversal.to_csv(os.path.join(output_dir, file), index=False)
            print(f'Exported to {output_dir}!')
        return LS_table_reversal

    def count_stock_of_traded(self):
        count_non_zero = (self.table['Long Short'] != 0).sum()
        proportion = count_non_zero
        return proportion
