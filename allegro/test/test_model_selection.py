import unittest

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from hyperopt import tpe, Trials
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from ..model_selection import (optimise_hyper_params, XGB_DEFAULT_SPACE,
                               LGB_DEFAULT_SPACE, CB_DEFAULT_SPACE)


class TestCv(unittest.TestCase):
    def setUp(self):
        reg_data = load_boston()
        cls_data = load_breast_cancer()
        self.reg_x_data = pd.DataFrame(reg_data.data)
        self.reg_y_data = reg_data.target
        self.cls_x_data = pd.DataFrame(cls_data.data)
        self.cls_y_data = cls_data.target

    def test_optimise_xgb(self):
        best = optimise_hyper_params(
            cls=xgb.XGBRegressor,
            X=self.reg_x_data,
            y=self.reg_y_data,
            estimator_params={'n_estimators': 100,
                              'learning_rate': 0.05,
                              'n_jobs': -1,
                              'random_state': 0},
            cv_params={'scoring': 'neg_mean_squared_error',
                       'cv': 5,
                       'greater_is_better': True},
            space=XGB_DEFAULT_SPACE,
            algo=tpe.suggest,
            max_evals=10,
            verbose=True
        )
        self.assertTrue(isinstance(best, Trials))

        best = optimise_hyper_params(
            cls=xgb.XGBClassifier,
            X=self.cls_x_data,
            y=self.cls_y_data,
            estimator_params={'n_estimators': 100,
                              'learning_rate': 0.05,
                              'n_jobs': -1,
                              'random_state': 0},
            cv_params={'scoring': 'accuracy',
                       'cv': 5,
                       'greater_is_better': True},
            space=XGB_DEFAULT_SPACE,
            algo=tpe.suggest,
            max_evals=10,
            verbose=True
        )
        self.assertTrue(isinstance(best, Trials))

    def test_optimise_lgb(self):
        best = optimise_hyper_params(
            cls=lgb.LGBMRegressor,
            X=self.reg_x_data,
            y=self.reg_y_data,
            estimator_params={'n_estimators': 100,
                              'learning_rate': 0.05,
                              'n_jobs': -1,
                              'random_state': 0,
                              'feature_fraction_seed': 0,
                              'bagging_seed': 0,
                              'verbose': -1},
            cv_params={'scoring': 'neg_mean_squared_error',
                       'cv': 5,
                       'greater_is_better': True},
            space=LGB_DEFAULT_SPACE,
            algo=tpe.suggest,
            max_evals=10,
            verbose=True
        )
        self.assertTrue(isinstance(best, Trials))

        best = optimise_hyper_params(
            cls=lgb.LGBMClassifier,
            X=self.cls_x_data,
            y=self.cls_y_data,
            estimator_params={'n_estimators': 100,
                              'learning_rate': 0.05,
                              'n_jobs': -1,
                              'random_state': 0,
                              'feature_fraction_seed': 0,
                              'bagging_seed': 0,
                              'verbose': -1},
            cv_params={'scoring': 'accuracy',
                       'cv': 5,
                       'greater_is_better': True},
            space=LGB_DEFAULT_SPACE,
            algo=tpe.suggest,
            max_evals=10,
            verbose=True
        )
        self.assertTrue(isinstance(best, Trials))

    def test_optimise_cb(self):
        best = optimise_hyper_params(
            cls=cb.CatBoostRegressor,
            X=self.reg_x_data,
            y=self.reg_y_data,
            estimator_params={'n_estimators': 100,
                              'learning_rate': 0.05,
                              'random_state': 0,
                              'verbose': False},
            cv_params={'scoring': 'neg_mean_squared_error',
                       'cv': 5,
                       'greater_is_better': True},
            space=CB_DEFAULT_SPACE,
            algo=tpe.suggest,
            max_evals=1,
            verbose=True
        )
        self.assertTrue(isinstance(best, Trials))

        best = optimise_hyper_params(
            cls=cb.CatBoostClassifier,
            X=self.cls_x_data,
            y=self.cls_y_data,
            estimator_params={'n_estimators': 100,
                              'learning_rate': 0.05,
                              'random_state': 0,
                              'verbose': False},
            cv_params={'scoring': 'accuracy',
                       'cv': 5,
                       'greater_is_better': True},
            space=CB_DEFAULT_SPACE,
            algo=tpe.suggest,
            max_evals=1,
            verbose=True
        )
        self.assertTrue(isinstance(best, Trials))

    def test_stratified_cv(self):
        best = optimise_hyper_params(
            cls=xgb.XGBClassifier,
            X=self.cls_x_data,
            y=self.cls_y_data,
            estimator_params={'n_estimators': 100,
                              'learning_rate': 0.05,
                              'n_jobs': -1,
                              'random_state': 0},
            cv_params={'scoring': 'accuracy',
                       'cv': StratifiedKFold(n_splits=5),
                       'greater_is_better': True},
            space=XGB_DEFAULT_SPACE,
            algo=tpe.suggest,
            max_evals=10,
            verbose=True
        )
        self.assertTrue(isinstance(best, Trials))
