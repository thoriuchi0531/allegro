from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb

from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)

from .log import get_logger

logger = get_logger(__name__)


def _xgb_plot_cv_result(cv_result):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    cv_result[['train-rmse-mean', 'test-rmse-mean']].plot(logy=True, ax=ax[0])
    cv_result[['train-rmse-std', 'test-rmse-std']].plot(logy=False, ax=ax[1])


def _lgb_plot_cv_result(cv_result):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    cv_result['rmse-mean'].plot(logy=True, ax=ax[0])
    cv_result['rmse-stdv'].plot(logy=False, ax=ax[1])


def _xgb_cv(x_train, y_train, **param_set):
    random_state = param_set.get('random_state', 0)
    xgb_model = xgb.XGBRegressor(n_estimators=5000, n_jobs=-1,
                                 random_state=random_state)
    xgb_model.set_params(**param_set)
    xgb_params = xgb_model.get_xgb_params()
    xgb_dmatrix = xgb.DMatrix(x_train, y_train)

    xgb_cv_result = xgb.cv(
        xgb_params,
        xgb_dmatrix,
        num_boost_round=xgb_params.get('n_estimators'),
        nfold=10,
        metrics='rmse',
        early_stopping_rounds=10,
        seed=random_state,
        verbose_eval=False
    )

    n_estimators = xgb_cv_result.shape[0]
    xgb_params['n_estimators'] = n_estimators
    xgb_model.set_params(n_estimators=n_estimators)
    return [xgb_params,
            n_estimators,
            xgb_cv_result.loc[n_estimators - 1, 'train-rmse-mean'],
            xgb_cv_result.loc[n_estimators - 1, 'train-rmse-std'],
            xgb_cv_result.loc[n_estimators - 1, 'test-rmse-mean'],
            xgb_cv_result.loc[n_estimators - 1, 'test-rmse-std'],
            xgb_cv_result,
            xgb_model]


def _lgb_cv(x_train, y_train, **param_set):
    random_state = param_set.get('random_state', 0)
    feature_fraction_seed = param_set.get('feature_fraction_seed',
                                          random_state)
    bagging_seed = param_set.get('bagging_seed', random_state)
    lgb_model = lgb.LGBMRegressor(n_estimators=5000, n_jobs=-1,
                                  random_state=random_state,
                                  feature_fraction_seed=feature_fraction_seed,
                                  bagging_seed=bagging_seed)
    lgb_model.set_params(**param_set)
    lgb_params = lgb_model.get_params()
    # Use .pop() to prevent warning
    n_estimators = lgb_params.pop('n_estimators')
    lgb_params.pop('silent')
    lgb_dataset = lgb.Dataset(x_train, label=y_train)

    lgb_cv_result = lgb.cv(
        lgb_params,
        lgb_dataset,
        num_boost_round=n_estimators,
        nfold=10,
        metrics='rmse',
        early_stopping_rounds=10,
        seed=random_state,
        verbose_eval=False,
        stratified=False
    )

    lgb_cv_result = pd.DataFrame(lgb_cv_result)
    n_estimators = lgb_cv_result.shape[0]
    lgb_params['n_estimators'] = n_estimators
    lgb_params['feature_fraction_seed'] = random_state
    lgb_params['bagging_seed'] = random_state
    lgb_model.set_params(n_estimators=n_estimators)
    return [lgb_params,
            n_estimators,
            None,
            None,
            lgb_cv_result.loc[n_estimators- 1, 'rmse-mean'],
            lgb_cv_result.loc[n_estimators- 1, 'rmse-stdv'],
            lgb_cv_result,
            lgb_model]


def _run_cv(x_train, y_train,
            func_get_n_estimators, func_plot_cv_result,
            plot_result=True, **kwargs):

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
        logger.info('param_list={}'.format(param_set))
        cv_result_last = func_get_n_estimators(x_train, y_train, **param_set)
        cv_summary.append(cv_result_last)

    test_rmse_mean = [i[4] for i in cv_summary]
    test_rmse_std = [i[5] for i in cv_summary]
    arg_min = np.argmin(test_rmse_mean)

    # show parameter sensitivity
    if len(test_rmse_mean) > 1 and plot_result:
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
    logger.info('best_params={}'.format(best_params))
    logger.info('\n{}'.format(best_cv_result
                              .iloc[best_params['n_estimators'] - 1, :]))
    if plot_result:
        func_plot_cv_result(best_cv_result)

    # fit best params
    estimator = cv_summary[arg_min][7]
    estimator.fit(x_train, y_train)
    return estimator


def run_xgb_cv(x_train, y_train, plot_result=True, **kwargs):
    return _run_cv(x_train, y_train,
                   _xgb_cv, _xgb_plot_cv_result,
                   plot_result, **kwargs)


def run_lgb_cv(x_train, y_train, plot_result=True, **kwargs):
    return _run_cv(x_train, y_train,
                   _lgb_cv, _lgb_plot_cv_result,
                   plot_result, **kwargs)
