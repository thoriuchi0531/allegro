import pandas as pd
from scipy.stats import ks_2samp


def run_ks_test(df1, df2):
    """ Run Kolmogorov-Smirnov test for each column
    H0: both follow the same distribution """
    if df1.shape[1] != df2.shape[1]:
        raise ValueError('Shape mismatch. df1={}, df2={}'
                         .format(df1.shape, df2.shape))

    ks_test_result = []
    for col_name in df1:
        ks_stat, p_value = ks_2samp(df1[col_name], df2[col_name])
        ks_test_result.append([
            col_name, ks_stat, p_value
        ])
    ks_test_result = pd.DataFrame(ks_test_result,
                                  columns=['col_name', 'ks_test', 'p_value'])
    return ks_test_result
