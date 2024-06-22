import numpy as np
from scipy import stats
from PCA_and_ETC import *

import statsmodels.api as sm


class metrics:
    def __init__(self, result_df, result_modified, period, total: bool):
        self.result_df = result_df
        self.result_modified = result_modified
        self.period = period
        self.total = total

    def cal_describe(self):
        for i in range(len(self.result_modified.columns)):
            self.result_modified.iloc[0, i] = len(self.result_df.columns)
            self.result_modified.iloc[1, i] = np.exp(np.mean(self.result_df.iloc[i, :]) * 12) - 1
            self.result_modified.iloc[2, i] = np.exp(np.std(self.result_df.iloc[i, :]) * np.sqrt(12)) - 1
            cum = (np.exp(sum(self.result_df.iloc[i, :])) - 1)*100
            self.result_modified.iloc[3, i] = cum

    def cal_annual_return(self, col):
        mean_return = pd.DataFrame(index=col, columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            for j in range(len(self.period)):
                row = self.result_df.iloc[i, self.period[j]]
                returns = np.exp(np.mean(row) * 12) - 1
                mean_return.iloc[j, i] = returns

        return mean_return

    def cal_sharpe_ratio(self, col):
        sharpe_ratio = pd.DataFrame(index=col, columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            for j in range(len(self.period)):
                row = self.result_df.iloc[i, self.period[j]]
                sf = (np.exp(np.mean(row) * 12) - 1) / (np.exp(row.values.std() * np.sqrt(12)) - 1)
                # sf = (np.mean(np.exp(row) - 1)) / (np.std(np.exp(row) - 1)) * np.sqrt(12)
                sharpe_ratio.iloc[j, i] = sf

        if self.total:
            return sharpe_ratio

        if not self.total:
            self.result_modified = pd.concat([self.result_modified, sharpe_ratio], axis=0)

    def cal_t_statistics(self):
        t_test = pd.DataFrame(index=['t-statistic'], columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            t_statistic, p_value = stats.ttest_ind_from_stats(self.result_modified.iloc[1, i],
                                                              self.result_modified.iloc[2, i],
                                                              self.result_modified.iloc[0, i],
                                                              self.result_modified.iloc[1, -2],
                                                              self.result_modified.iloc[2, -2],
                                                              self.result_modified.iloc[0, -2])
            t_test.iloc[0, i] = t_statistic

        self.result_modified = pd.concat([self.result_modified, t_test], axis=0)

    def cal_down_std(self, col):
        Down_std = pd.DataFrame(index=col, columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            for j in range(len(self.period)):
                row = self.result_df.iloc[i, self.period[j]]

                # row = self.result_df.iloc[i, :]
                if len(row[row < 0]) <= 1:
                    dd = 0.01

                else:
                    dd = (np.exp(np.mean(np.sqrt(np.square(row[row < 0]))) * np.sqrt(12)) - 1)

                Down_std.iloc[j, i] = dd

        if self.total:
            return Down_std

        if not self.total:
            self.result_modified = pd.concat([self.result_modified, Down_std], axis=0)

    def cal_sortino_ratio(self, col):
        sortino_ratio = pd.DataFrame(index=col, columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            for j in range(len(self.period)):
                row = self.result_df.iloc[i, self.period[j]]
                # sf = (np.exp(row.mean() * 12) - 1) / (np.exp(row[row < 0].std() * np.sqrt(12)) - 1)
                if len(row[row < 0]) <= 1:
                    dd = 0.01

                else:
                    dd = np.exp(np.std(np.sqrt(np.square(row[row < 0]))) * np.sqrt(12)) - 1

                sf = ((np.exp(np.mean(row) * 12) - 1) / dd)
                sortino_ratio.iloc[j, i] = sf

        if self.total:
            return sortino_ratio

        if not self.total:
            self.result_modified = pd.concat([self.result_modified, sortino_ratio], axis=0)

    def cal_gross_profit(self):
        Gross_profit = pd.DataFrame(index=['Gross profit'], columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            row = self.result_df.iloc[i, :]
            pf = row[row > 0].sum()
            Gross_profit.iloc[0, i] = pf

        self.result_modified = pd.concat([self.result_modified, Gross_profit], axis=0)

    def cal_gross_loss(self):
        Gross_loss = pd.DataFrame(index=['Gross loss'], columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            row = self.result_df.iloc[i, :]
            pf = row[row < 0].sum()
            Gross_loss.iloc[0, i] = pf

        self.result_modified = pd.concat([self.result_modified, Gross_loss], axis=0)

    def cal_profit_factor(self, col):
        profit_factor = pd.DataFrame(index=col, columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            for j in range(len(self.period)):
                row = self.result_df.iloc[i, self.period[j]]
                pf = row[row > 0].sum() / np.abs(row[row < 0].sum())
                profit_factor.iloc[j, i] = pf

        if self.total:
            return profit_factor

        if not self.total:
            self.result_modified = pd.concat([self.result_modified, profit_factor], axis=0)

    def cal_profitable_years(self):
        profitable_years = pd.DataFrame(index=['Profitable years'], columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            row = self.result_df.iloc[i, :]
            row.index = pd.to_datetime(row.index)
            row2 = pd.DataFrame(row.index.year, index=row.index)
            row = pd.concat([row, row2], axis=1)
            sum_by_year = row.groupby(row.index.year)[self.result_modified.columns[i]].sum()
            pf = sum_by_year[sum_by_year > 0].count()
            profitable_years.iloc[0, i] = pf

        self.result_modified = pd.concat([self.result_modified, profitable_years], axis=0)

    def cal_unprofitable_years(self):
        unprofitable_years = pd.DataFrame(index=['Unprofitable years'], columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            row = self.result_df.iloc[i, :]
            row.index = pd.to_datetime(row.index)
            row2 = pd.DataFrame(row.index.year, index=row.index)
            row = pd.concat([row, row2], axis=1)
            sum_by_year = row.groupby(row.index.year)[self.result_modified.columns[i]].sum()
            pf = sum_by_year[sum_by_year < 0].count()
            unprofitable_years.iloc[0, i] = pf

        self.result_modified = pd.concat([self.result_modified, unprofitable_years], axis=0)

    def cal_MDD(self, col):
        MDD = pd.DataFrame(index=col, columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            for j in range(len(self.period)):
                row = self.result_df.iloc[i, self.period[j]]
                cumulative_returns = np.exp(np.cumsum(row))
                max_drawdown = 0
                peak = 1  # 초기 최대 누적 수익은 1 (시작값)

                # 각 시점에서 최대 손실 계산
                for k in range(len(cumulative_returns)):
                    if cumulative_returns[k] > peak:
                        peak = cumulative_returns[k]
                    else:
                        drawdown = (peak - cumulative_returns[k]) / peak
                        max_drawdown = max(max_drawdown, drawdown)

                MDD.iloc[j, i] = -max_drawdown

        if self.total:
            return MDD

        if not self.total:
            self.result_modified = pd.concat([self.result_modified, MDD], axis=0)

    def cal_calmar_ratio(self, col):
        Calmar_ratio = pd.DataFrame(index=col, columns=self.result_modified.columns)
        for i in range(len(self.result_modified.columns)):
            for j in range(len(self.period)):
                row = self.result_df.iloc[i, self.period[j]]
                row2 = np.exp(row.astype(float)) - 1
                cumulative_returns = np.cumprod(1 + row2) - 1
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = ((cumulative_returns - peak) / (peak))
                max_drawdown = drawdown.min()
                calmar = (np.mean(np.exp(row) - 1) * 12) / abs(max_drawdown)
                Calmar_ratio.iloc[j, i] = calmar

        if self.total:
            return Calmar_ratio

        if not self.total:
            self.result_modified = pd.concat([self.result_modified, Calmar_ratio], axis=0)

    def cal_monthly_statistics(self):
        month = pd.DataFrame(
            index=['Mean', 'Standard deviation', 'Standard error', 't-statistic', 'Min', '25%', '50%', '75%', 'Max',
                   'Skew', 'Kurtosis'], columns=self.result_modified.columns)

        for i in range(len(self.result_modified.columns)):
            month.iloc[0, i] = np.mean(self.result_df.iloc[i, :])
            month.iloc[1, i] = np.std(self.result_df.iloc[i, :])
            month.iloc[2, i] = np.std(self.result_df.iloc[i, :], ddof=1) / np.sqrt(len(self.result_df.iloc[i, :]))
            month.iloc[4, i] = np.min(self.result_df.iloc[i, :])
            month.iloc[5, i] = np.percentile(self.result_df.iloc[i, :], 25)
            month.iloc[6, i] = np.percentile(self.result_df.iloc[i, :], 50)
            month.iloc[7, i] = np.percentile(self.result_df.iloc[i, :], 75)
            month.iloc[8, i] = np.max(self.result_df.iloc[i, :])
            month.iloc[9, i] = self.result_df.iloc[i, :].skew()
            month.iloc[10, i] = self.result_df.iloc[i, :].kurtosis()

            X = sm.add_constant(self.result_df.iloc[i, :].shift(1).dropna())
            y = self.result_df.iloc[i, :][1:].dropna()
            y.index = X.index

            model = sm.OLS(y, X)
            newey_west = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
            t_statistic, p_value = newey_west.tvalues
            month.iloc[3, i] = t_statistic

        self.result_modified = pd.concat([self.result_modified, month], axis=0)
