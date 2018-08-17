from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from xgboost import XGBRegressor

from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)


def plot_cv_result(cv_result):
    fix, ax = plt.subplots(1, 2, figsize=(14, 4))
    cv_result[['train-rmse-mean', 'test-rmse-mean']].plot(logy=True, ax=ax[0])
    cv_result[['train-rmse-std', 'test-rmse-std']].plot(logy=False, ax=ax[1])


def run_xgb_cv(x_train, y_train, **kwargs):
    def _run_xgb_cv(**param_set):
        xgb_model = XGBRegressor(n_estimators=5000, n_jobs=-1)
        xgb_model.set_params(**param_set)
        xgb_params = xgb_model.get_xgb_params()
        xgb_dmatrix = xgboost.DMatrix(x_train, y_train)

        xgb_cv_result = xgboost.cv(
            xgb_params,
            xgb_dmatrix,
            num_boost_round=xgb_params.get('n_estimators'),
            nfold=10,
            metrics='rmse',
            early_stopping_rounds=10,
            verbose_eval=False
        )

        n_estimators = xgb_cv_result.shape[0]
        xgb_params['n_estimators'] = n_estimators
        return [xgb_params,
                n_estimators,
                xgb_cv_result.loc[n_estimators - 1, 'train-rmse-mean'],
                xgb_cv_result.loc[n_estimators - 1, 'train-rmse-std'],
                xgb_cv_result.loc[n_estimators - 1, 'test-rmse-mean'],
                xgb_cv_result.loc[n_estimators - 1, 'test-rmse-std'],
                xgb_cv_result]

    param_list = dict()
    for kw_k, kw_v in kwargs.items():
        if isinstance(kw_v, (tuple, list)):
            param_list[kw_k] = kw_v
        else:
            param_list[kw_k] = [kw_v]

    cv_summary = []
    for param_values in tqdm_notebook(list(product(*param_list.values()))):
        # this will be executed even if kwargs is None
        param_set = {k: v for k, v in zip(param_list.keys(), param_values)}
        print('param_list={}'.format(param_set))
        xgb_cv_result_last = _run_xgb_cv(**param_set)
        cv_summary.append(xgb_cv_result_last)

    test_rmse_mean = [i[4] for i in cv_summary]
    test_rmse_std = [i[5] for i in cv_summary]
    arg_min = np.argmin(test_rmse_mean)

    # show parameter sensitivity
    if len(test_rmse_mean) > 1:
        all_params = [i[0] for i in cv_summary]
        concat_params = pd.concat([pd.Series(i) for i in all_params], axis=1)
        concat_params = concat_params[concat_params.nunique(axis=1) > 1]
        all_params = [concat_params[i].to_dict() for i in concat_params]
        sensi_mean = pd.Series(test_rmse_mean, index=all_params)
        sensi_std = pd.Series(test_rmse_std, index=all_params)
        sensi_mean.to_frame().plot(kind='barh', xerr=sensi_std, figsize=(8, 6))

    # show result on the best CV result
    best_params = cv_summary[arg_min][0]
    best_cv_result = cv_summary[arg_min][6]
    print('best_params={}'.format(best_params))
    print(best_cv_result.iloc[best_params['n_estimators'] - 1, :])
    plot_cv_result(best_cv_result)

    # fit best params
    xgb_model = XGBRegressor(n_estimators=5000, n_jobs=-1)
    xgb_model.set_params(**best_params)
    xgb_model.fit(x_train, y_train)

    return xgb_model
