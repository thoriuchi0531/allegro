from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

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


def _cb_plot_cv_result(cv_result):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    cv_result[['train-RMSE-mean', 'test-RMSE-mean']].plot(logy=True, ax=ax[0])
    cv_result[['train-RMSE-std', 'test-RMSE-std']].plot(logy=False, ax=ax[1])


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
    lgb_model = lgb.LGBMRegressor(n_estimators=5000,
                                  seed=random_state,
                                  feature_fraction_seed=feature_fraction_seed,
                                  bagging_seed=bagging_seed,
                                  verbose=-1)
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
            lgb_cv_result.loc[n_estimators - 1, 'rmse-mean'],
            lgb_cv_result.loc[n_estimators - 1, 'rmse-stdv'],
            lgb_cv_result,
            lgb_model]


def _cb_cv(x_train, y_train, **param_set):
    random_state = param_set.get('random_state', 0)
    cb_model = cb.CatBoostRegressor(n_estimators=5000, loss_function='RMSE',
                                    random_state=random_state,
                                    verbose=False)
    cb_model.set_params(**param_set)
    cb_params = cb_model.get_params()
    n_estimators = cb_params.pop('iterations')
    cb_pool = cb.Pool(x_train, label=y_train)

    cb_cv_result = cb.cv(
        pool=cb_pool,
        params=cb_params,
        num_boost_round=n_estimators,
        nfold=10,
        early_stopping_rounds=10,
        seed=random_state,
        verbose_eval=False,
        stratified=False,
    )
    n_estimators = cb_cv_result.shape[0] - 10
    cb_params['n_estimators'] = n_estimators
    cb_model.set_params(iterations=n_estimators)
    return [cb_params,
            n_estimators,
            cb_cv_result.loc[n_estimators - 1, 'train-RMSE-mean'],
            cb_cv_result.loc[n_estimators - 1, 'train-RMSE-std'],
            cb_cv_result.loc[n_estimators - 1, 'test-RMSE-mean'],
            cb_cv_result.loc[n_estimators - 1, 'test-RMSE-std'],
            cb_cv_result,
            cb_model]


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


def run_cb_cv(x_train, y_train, plot_result=True, **kwargs):
    return _run_cv(x_train, y_train,
                   _cb_cv, _cb_plot_cv_result,
                   plot_result, **kwargs)


def run_xgb_optimise(X, y, plot_result=False, **xgb_params):
    def _log_header(text):
        logger.info('\n------------------------------------------------------\n'
                    '\t {}\n'
                    '------------------------------------------------------'
                    .format(text))

    def _cv_xgb(X, y, max_depth_list, min_child_weight_list):
        xgb_params.update({
            'max_depth': max_depth_list,
            'min_child_weight': min_child_weight_list,
        })
        model = run_xgb_cv(X, y, plot_result=plot_result, **xgb_params)
        max_depth = model.get_xgb_params()['max_depth']
        min_child_weight = model.get_xgb_params()['min_child_weight']
        return max_depth, min_child_weight

    _log_header('max_depth and min_child_weight')
    max_depth_list = list(range(1, 10, 2))
    min_child_weight_list = list(range(1, 10, 2))
    max_depth_list2 = list(range(9, 20, 2))
    min_child_weight_list2 = list(range(9, 20, 2))

    max_depth, min_child_weight = _cv_xgb(X, y, max_depth_list,
                                          min_child_weight_list)

    if (max_depth < max(max_depth_list) and
            min_child_weight < max(min_child_weight_list)):
        logger.info('Trying finer grids')
        max_depth, min_child_weight = _cv_xgb(
            X, y,
            [max_depth - 1, max_depth, max_depth + 1],
            [min_child_weight - 1, min_child_weight, min_child_weight + 1]
        )

    elif (max_depth >= max(max_depth_list) and
          min_child_weight < max(min_child_weight_list)):
        logger.info('Optimal max_depth may be outside of the initial range.')
        max_depth, min_child_weight = _cv_xgb(
            X, y,
            max_depth_list2,
            [min_child_weight - 1, min_child_weight, min_child_weight + 1]
        )

        logger.info('Trying finer grids for max_depth')
        max_depth, min_child_weight = _cv_xgb(
            X, y,
            [max_depth - 1, max_depth, max_depth + 1],
            min_child_weight
        )

    elif (max_depth < max(max_depth_list) and
          min_child_weight >= max(min_child_weight_list)):
        logger.info(
            'Optimal min_child_weight may be outside of the initial range.')
        max_depth, min_child_weight = _cv_xgb(
            X, y,
            [max_depth - 1, max_depth, max_depth + 1],
            min_child_weight_list2
        )

        logger.info('Trying finer grids for min_child_weight')
        max_depth, min_child_weight = _cv_xgb(
            X, y,
            max_depth,
            [min_child_weight - 1, min_child_weight, min_child_weight + 1]
        )

    else:
        logger.info('Optimal max_depth may be outside of the initial range.')
        logger.info(
            'Optimal min_child_weight may be outside of the initial range.')
        max_depth, min_child_weight = _cv_xgb(X, y, max_depth_list2,
                                              min_child_weight_list2)

        if (max_depth > max(max_depth_list2) or
                min_child_weight > max(min_child_weight_list2)):
            raise ValueError('Too large max_depth or min_child_weight. '
                             'max_depth={}, min_child_weight={}'
                             .format(max_depth, min_child_weight))

        logger.info('Trying finer grids')
        max_depth, min_child_weight = _cv_xgb(
            X, y,
            [max_depth - 1, max_depth, max_depth + 1],
            [min_child_weight - 1, min_child_weight, min_child_weight + 1]
        )

    # --------------------------------------------------------------------------
    # gamma
    # --------------------------------------------------------------------------
    _log_header('gamma')
    xgb_params.update({
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'gamma': [i / 10 for i in range(0, 10)],
    })
    model = run_xgb_cv(X, y, plot_result=plot_result, **xgb_params)
    gamma = model.get_xgb_params()['gamma']

    # --------------------------------------------------------------------------
    # subsample and colsample_bytree
    # --------------------------------------------------------------------------
    _log_header('subsample and colsample_bytree')
    subsample_list = [i / 10 for i in range(1, 11, 2)] + [1.0]
    colsample_bytree_list = [i / 10 for i in range(1, 11, 2)] + [1.0]
    xgb_params.update({
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'subsample': subsample_list,
        'colsample_bytree': colsample_bytree_list
    })
    model = run_xgb_cv(X, y, plot_result=plot_result, **xgb_params)
    subsample = model.get_xgb_params()['subsample']
    colsample_bytree = model.get_xgb_params()['colsample_bytree']

    if subsample in [0.1, 1.0] and colsample_bytree in [0.1, 1.0]:
        # already optimal
        pass
    else:
        logger.info('Trying finer grids')
        subsample_list = [subsample - 0.1, subsample, subsample + 0.1]
        colsample_bytree_list = [colsample_bytree - 0.1, colsample_bytree,
                                 colsample_bytree + 0.1]
        if subsample == 0.1:
            subsample_list = [0.1]
        elif subsample == 1.0:
            subsample_list = [1.0]
        if colsample_bytree == 0.1:
            colsample_bytree_list = [0.1]
        elif colsample_bytree == 1.0:
            colsample_bytree_list = [1.0]
        xgb_params.update({
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample_list,
            'colsample_bytree': colsample_bytree_list
        })
        model = run_xgb_cv(X, y, plot_result=plot_result, **xgb_params)
        subsample = model.get_xgb_params()['subsample']
        colsample_bytree = model.get_xgb_params()['colsample_bytree']

    # --------------------------------------------------------------------------
    # reg_alpha
    # --------------------------------------------------------------------------
    _log_header('reg_alpha')
    xgb_params.update({
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 10, 100],
    })
    model = run_xgb_cv(X, y, plot_result=plot_result, **xgb_params)
    reg_alpha = model.get_xgb_params()['reg_alpha']

    # --------------------------------------------------------------------------
    # reg_lambda
    # --------------------------------------------------------------------------
    _log_header('reg_lambda')
    xgb_params.update({
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 10, 100],
    })
    model = run_xgb_cv(X, y, plot_result=plot_result, **xgb_params)
    reg_lambda = model.get_xgb_params()['reg_lambda']

    # --------------------------------------------------------------------------
    # learning_rate
    # --------------------------------------------------------------------------
    _log_header('learning_rate')
    xgb_params.update({
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'learning_rate': [.3, .2, .1, .05, .01, .005]
    })
    model = run_xgb_cv(X, y, plot_result=plot_result, **xgb_params)

    return model


def run_lgb_optimise(X, y, plot_result=False, **lgb_params):
    def _log_header(text):
        logger.info('\n------------------------------------------------------\n'
                    '\t {}\n'
                    '------------------------------------------------------'
                    .format(text))

    # --------------------------------------------------------------------------
    # num_leaves and max_bin
    # --------------------------------------------------------------------------
    _log_header('num_leaves and max_bin')
    lgb_params.update({
        'num_leaves': list(range(2, 6)) + list(range(6, 50, 5)),
        'max_bin': [8, 16, 32, 64, 128, 255, 512]
    })
    model = run_lgb_cv(X, y, plot_result=plot_result, **lgb_params)
    num_leaves = model.get_params()['num_leaves']
    max_bin = model.get_params()['max_bin']

    logger.info('Trying finer grids')
    lgb_params.update({
        'num_leaves': num_leaves,
        'max_bin': [int(max_bin / (2 ** 0.5)), max_bin,
                    int(max_bin * (2 ** 0.5))]
    })
    model = run_lgb_cv(X, y, plot_result=plot_result, **lgb_params)
    num_leaves = model.get_params()['num_leaves']
    max_bin = model.get_params()['max_bin']

    # --------------------------------------------------------------------------
    # max_depth
    # --------------------------------------------------------------------------
    _log_header('max_depth')
    lgb_params.update({
        'num_leaves': num_leaves,
        'max_bin': max_bin,
        'max_depth': list(range(1, 11))
    })
    model = run_lgb_cv(X, y, plot_result=plot_result, **lgb_params)
    max_depth = model.get_params()['max_depth']

    # --------------------------------------------------------------------------
    # min_data_in_leaf and min_sum_hessian_in_leaf
    # --------------------------------------------------------------------------
    _log_header('min_data_in_leaf and min_sum_hessian_in_leaf')
    lgb_params.update({
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'max_bin': max_bin,
        'min_data_in_leaf': list(range(5, 40, 5)),
        'min_sum_hessian_in_leaf': [1e-5, 1e-4, 1e-3, 1e-2]
    })
    model = run_lgb_cv(X, y, plot_result=plot_result, **lgb_params)
    min_data_in_leaf = model.get_params()['min_data_in_leaf']
    min_sum_hessian_in_leaf = model.get_params()['min_sum_hessian_in_leaf']

    # --------------------------------------------------------------------------
    # bagging_fraction and bagging_freq
    # --------------------------------------------------------------------------
    _log_header('bagging_fraction and bagging_freq')
    bagging_fraction_list = [i / 10 for i in range(1, 11, 2)] + [1.0]
    bagging_freq_list = [1, 5, 10]

    lgb_params.update({
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'max_bin': max_bin,
        'min_data_in_leaf': min_data_in_leaf,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': bagging_fraction_list,
        'bagging_freq': bagging_freq_list,
    })
    model = run_lgb_cv(X, y, plot_result=plot_result, **lgb_params)
    bagging_fraction = model.get_params()['bagging_fraction']
    bagging_freq = model.get_params()['bagging_freq']

    # --------------------------------------------------------------------------
    # feature_fraction
    # --------------------------------------------------------------------------
    _log_header('feature_fraction')
    feature_fraction_list = [i / 10 for i in range(1, 11)]

    lgb_params.update({
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'max_bin': max_bin,
        'min_data_in_leaf': min_data_in_leaf,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'feature_fraction': feature_fraction_list,
    })
    model = run_lgb_cv(X, y, plot_result=plot_result, **lgb_params)
    feature_fraction = model.get_params()['feature_fraction']

    # --------------------------------------------------------------------------
    # reg_alpha
    # --------------------------------------------------------------------------
    _log_header('reg_alpha')
    lgb_params.update({
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'max_bin': max_bin,
        'min_data_in_leaf': min_data_in_leaf,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'feature_fraction': feature_fraction,
        'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],
    })
    model = run_lgb_cv(X, y, plot_result=plot_result, **lgb_params)
    reg_alpha = model.get_params()['reg_alpha']

    # --------------------------------------------------------------------------
    # reg_lambda
    # --------------------------------------------------------------------------
    _log_header('reg_lambda')
    lgb_params.update({
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'max_bin': max_bin,
        'min_data_in_leaf': min_data_in_leaf,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'feature_fraction': feature_fraction,
        'reg_alpha': reg_alpha,
        'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],
    })
    model = run_lgb_cv(X, y, plot_result=plot_result, **lgb_params)
    reg_lambda = model.get_params()['reg_lambda']

    # --------------------------------------------------------------------------
    # learning_rate
    # --------------------------------------------------------------------------
    _log_header('learning_rate')
    lgb_params.update({
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'max_bin': max_bin,
        'min_data_in_leaf': min_data_in_leaf,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'feature_fraction': feature_fraction,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'learning_rate': [.3, .2, .1, .05, .01, .005]
    })
    model = run_lgb_cv(X, y, plot_result=plot_result, **lgb_params)

    return model