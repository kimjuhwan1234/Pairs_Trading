from utils.PCA_and_ETC import *
from multiprocessing import Pool
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint, kpss

import statsmodels.api as sm

warnings.filterwarnings("ignore")


class cointegration:

    def __init__(self, output_dir, file, data):
        self.data = data
        self.invest_list = []
        self.output_dir = output_dir
        self.file = file

    def read_mom_data(self):
        # mom1 save and data Normalization
        mom1 = self.data[momentum_prefix_finder(self.data) + '1']

        Scaler = StandardScaler()
        data_scaled = Scaler.fit_transform(self.data)

        # mom1을 제외한 mat/PCA(2-49)
        # df_combined = data_scaled.drop(columns=momentum_prefix_finder(data_scaled) + '1')

        # mom49를 제외한 mat/PCA(1-48)
        df_combined = data_scaled.drop(columns=momentum_prefix_finder(data_scaled) + '49')

        df_combined.insert(0, '0', mom1)
        df_combined = df_combined.dropna(axis=1, how='any')

        self.data = df_combined.T

    def cointegrate(self, pairs):
        x = self.data[pairs[0]].values
        y = self.data[pairs[1]].values
        _, p_value, _ = coint(x, y)
        return p_value

    def adf_result(self, value):
        try:
            ret = sm.tsa.adfuller(value)[1]
        except ValueError:
            ret = 0.06
        return ret

    def kpss_result(self, value):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                ret = kpss(value)[1]
        except Exception:
            ret = 0.04
        return ret

    def find_cointegrated_pairs(self):

        data = self.data.iloc[1:, :]
        pairs = pd.DataFrame(combinations(data.columns, 2))  # 모든 회사 조합

        with Pool(processes=5) as pool:
            pairs['pvalue'] = pool.map(self.cointegrate, pairs.values)
        pairs = pd.DataFrame(pairs.loc[pairs.index[pairs['pvalue'] < 0.01], :])
        print('Finished filtering pairs using pvalue!')

        spread_df: pd.DataFrame = pairs.apply(lambda x: data[x[0]] - data[x[1]], axis=1)

        with Pool(processes=5) as pool:
            spread_df['adf_result'] = pool.map(self.adf_result, spread_df.values)
        spread_df = pd.DataFrame(spread_df.loc[spread_df.index[spread_df['adf_result'] < 0.05], :])
        print('Finished filtering pairs using adf_result!')

        with Pool(processes=5) as pool:
            spread_df['kpss_result'] = pool.map(self.kpss_result, spread_df.values)
        spread_df = spread_df.loc[spread_df.index[spread_df['kpss_result'] > 0.05], :]
        print('Finished filtering pairs using kpss_result!')

        spread_df = spread_df.drop(columns=['adf_result'])
        spread_df = spread_df.drop(columns=['kpss_result'])

        if spread_df.empty:
            return []

        spread_sr = spread_df[momentum_prefix_finder(spread_df) + '1']
        pairs['spread'] = (spread_sr - spread_sr.mean()) / spread_sr.std()
        pairs = pd.DataFrame(pairs.dropna(subset=['spread']))
        pairs = pairs.loc[pairs.index[pairs['spread'].abs() > 2], :]
        pairs['pair1'] = pairs[0] * (pairs['spread'] > 0) + pairs[1] * (pairs['spread'] <= 0)
        pairs['pair2'] = pairs[0] * (pairs['spread'] <= 0) + pairs[1] * (pairs['spread'] > 0)
        pairs = pairs.drop(columns=[0, 1])
        print('Finished filtering pairs using normalised spread!')

        pairs.sort_values(by='pvalue', axis=0, inplace=True)
        pairs = pairs.drop_duplicates(subset='pair1')
        pairs = pairs.drop_duplicates(subset='pair2')
        invest_list = pd.DataFrame(pairs.values.tolist())
        invest_list = invest_list.iloc[:, 2:4]
        invest_list = invest_list.values.tolist()

        self.invest_list = invest_list

    def save_cointegrated_LS(self):
        LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'L_Result S_Result', 'Cluster Index'])
        # name = momentum_prefix_finder(self.data.T) + '1'

        name = '0'

        for cluster_num, firms in enumerate(self.invest_list):
            # Sort firms based on momentum_1
            long_short = [0] * 2
            long_short[0] = -1
            long_short[1] = 1

            # Add the data to the new table
            for i, firm in enumerate(firms):
                LS_table.loc[len(LS_table)] = [firm, self.data.T.loc[firm, name], long_short[i], cluster_num]

        firm_list_after = list(LS_table['Firm Name'])
        firm_list_before = list(self.data.T.index)
        Missing = [item for item in firm_list_before if item not in firm_list_after]

        for i, firm in enumerate(Missing):
            LS_table.loc[len(LS_table)] = [firm, self.data.T.loc[firm, name], 0, -1]

        LS_table.sort_values(by='Cluster Index', inplace=True)
        LS_table = LS_table[~LS_table.iloc[:, 0].duplicated(keep='first')]

        # Save the output to a CSV file in the output directory
        LS_table.to_csv(os.path.join(self.output_dir, self.file), index=False)
