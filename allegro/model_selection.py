from hyperopt import hp
from hyperopt.fmin import fmin
from hyperopt.pyll.base import scope
from sklearn.model_selection import cross_val_score

from .log import get_logger

logger = get_logger(__name__)

XGB_DEFAULT_SPACE = {
    'max_depth': scope.int(hp.quniform('max_depth', 2, 10, 1)),
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 2, 10, 1)),
    'gamma': hp.quniform('gamma', 0.0, 0.5, 0.1),
    'subsample': hp.quniform('subsample', 0.5, 1.0, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1.0, 0.1),
    'reg_alpha': hp.loguniform('reg_alpha', -10, 5),
    'reg_lambda': hp.loguniform('reg_lambda', -10, 5),
}

LGB_DEFAULT_SPACE = {
    'num_leaves': scope.int(hp.quniform('num_leaves', 2, 50, 1)),
    'max_bin': scope.int(hp.qloguniform('max_bin', 1, 7, 1)),
    'max_depth': scope.int(hp.quniform('max_depth', 2, 10, 1)),
    'min_data_in_leaf': scope.int(hp.quniform('min_data_in_leaf', 5, 50, 5)),
    'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -12, -3),
    'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1.0, 0.1),
    'bagging_freq': hp.choice('bagging_freq', [1, 5, 10]),
    'feature_fraction': hp.quniform('feature_fraction', 0.5, 1.0, 0.1),
    'reg_alpha': hp.loguniform('reg_alpha', -10, 5),
    'reg_lambda': hp.loguniform('reg_lambda', -10, 5),
}

CB_DEFAULT_SPACE = {
    'depth': scope.int(hp.quniform('depth', 2, 10, 1)),
    'border_count': hp.qloguniform('border_count', 1, 5, 1),
    'l2_leaf_reg': hp.loguniform('l2_leaf_reg', -10, 5),
    'bagging_temperature': hp.loguniform('bagging_temperature', -4, 4),
    'random_strength': hp.loguniform('random_strength', -4, 4),
}


def optimise_hyper_params(cls, X, y, estimator_params, cv_params, space, algo,
                          max_evals, verbose=False):
    def objective(params):
        # In sklearn, higher score values are better.
        greater_is_better = cv_params.pop('greater_is_better', True)

        estimator = cls(**estimator_params, **params)
        raw_score = cross_val_score(estimator, X, y, **cv_params).mean()
        if verbose:
            logger.info("Score {}, params {}".format(raw_score, params))
        if greater_is_better:
            # hyperopt minimises the score
            score = - raw_score
        else:
            score = raw_score
        return score

    best = fmin(fn=objective,
                space=space,
                algo=algo,
                max_evals=max_evals)
    return best
