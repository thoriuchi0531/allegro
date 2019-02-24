import unittest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_boston, load_breast_cancer
from ..pipeline import (FilterByImportance, ConvertNaNs, FillNa, GroupFillNa,
                        ConditionalFillNa, ConvertStrToInt,
                        ClassifierEnsemble, RegressorEnsemble)
import xgboost as xgb
import lightgbm as lgb


class TestPipeline(unittest.TestCase):
    def setUp(self):
        reg_data = load_boston()
        cls_data = load_breast_cancer()
        self.reg_x_data = pd.DataFrame(reg_data.data)
        self.reg_y_data = reg_data.target
        self.cls_x_data = pd.DataFrame(cls_data.data)
        self.cls_y_data = cls_data.target

    def test_filter_xgb_importance_threshold(self):
        """ Filter based on threshold """
        estimator = xgb.XGBRegressor()
        pipeline = FilterByImportance(estimator=estimator, threshold='mean')
        pipeline.fit(self.reg_x_data, self.reg_y_data)
        filtered = pipeline.transform(self.reg_x_data)

        importance = pipeline.estimator_.feature_importances_
        importance_mean = np.mean(importance)
        flg = importance > importance_mean
        self.assertTrue(set(self.reg_x_data.columns[flg]), set(filtered.columns))

    def test_filter_xgb_importance_n_features(self):
        """ Filter based on the number of features """
        estimator = xgb.XGBRegressor()
        pipeline = FilterByImportance(estimator=estimator, n_features=3)
        pipeline.fit(self.reg_x_data, self.reg_y_data)
        filtered = pipeline.transform(self.reg_x_data)

        importance = pipeline.estimator_.feature_importances_
        importance_rank = np.argsort(importance)
        flg = importance_rank >= len(self.reg_x_data.columns) - pipeline.n_features
        self.assertTrue(set(self.reg_x_data.columns[flg]), set(filtered.columns))

    def test_filter_lgb_importance_threshold(self):
        """ Filter based on threshold """
        estimator = lgb.LGBMRegressor(verbose=-1)
        pipeline = FilterByImportance(estimator=estimator, threshold='mean')
        pipeline.fit(self.reg_x_data, self.reg_y_data)
        filtered = pipeline.transform(self.reg_x_data)

        importance = pipeline.estimator_.feature_importances_
        importance_mean = np.mean(importance)
        flg = importance > importance_mean
        self.assertTrue(set(self.reg_x_data.columns[flg]), set(filtered.columns))

    def test_filter_lgb_importance_n_features(self):
        """ Filter based on the number of features """
        estimator = lgb.LGBMRegressor(verbose=-1)
        pipeline = FilterByImportance(estimator=estimator, n_features=3)
        pipeline.fit(self.reg_x_data, self.reg_y_data)
        filtered = pipeline.transform(self.reg_x_data)

        importance = pipeline.estimator_.feature_importances_
        importance_rank = np.argsort(importance)
        flg = importance_rank >= len(self.reg_x_data.columns) - pipeline.n_features
        self.assertTrue(set(self.reg_x_data.columns[flg]), set(filtered.columns))

    def test_convert_nan(self):
        X = pd.DataFrame([[np.nan, 2], [6, np.nan], [7, 6]],
                         columns=['a', 'b'])
        result = ConvertNaNs(target_columns=float).fit_transform(X)
        self.assertEqual(result.loc[0, 'a'], 6.5)
        self.assertEqual(result.loc[1, 'b'], 4)

    def test_fill_na(self):
        X = pd.DataFrame([[np.nan, 2], [6, np.nan], [7, 6]],
                         columns=['a', 'b'])
        result = FillNa(strategy=1, target_columns='a').fit_transform(X)
        self.assertEqual(result.loc[0, 'a'], 1)
        self.assertTrue(np.isnan(result.loc[1, 'b']))

        result = (FillNa(strategy=[10, 20], target_columns=['a', 'b'])
                  .fit_transform(X))
        self.assertEqual(result.loc[0, 'a'], 10)
        self.assertEqual(result.loc[1, 'b'], 20)

    def test_group_fill_na(self):
        X = pd.DataFrame([[np.nan, 2],
                          [6, np.nan],
                          [7, 6],
                          [np.nan, np.nan]],
                         columns=['a', 'b'])
        result = (GroupFillNa(strategy=[1, 2], target_columns=['a', 'b'])
                  .fit_transform(X))
        self.assertTrue(np.isnan(result.loc[0, 'a']))
        self.assertTrue(np.isnan(result.loc[1, 'b']))
        self.assertEqual(result.loc[3, 'a'], 1)
        self.assertEqual(result.loc[3, 'b'], 2)

    def test_conditional_fill_na(self):
        X = pd.DataFrame([['foo', 2],
                          ['bar', 1],
                          ['foo', 6],
                          ['bar', 10],
                          ['foo', np.nan],
                          ['bar', np.nan]],
                         columns=['a', 'b'])
        fit = (ConditionalFillNa(target_column='b', cond_column='a',
                                 how='mean').fit(X))
        result = fit.transform(X)
        self.assertTrue(result.loc[4, 'b'], 4)
        self.assertTrue(result.loc[5, 'b'], 5.5)

        X = pd.DataFrame([['foo', np.nan],
                          ['bar', np.nan]],
                         columns=['a', 'b'])
        result = fit.transform(X)
        self.assertTrue(result.loc[0, 'b'], 4)
        self.assertTrue(result.loc[1, 'b'], 5.5)

    def test_convert_str_to_int(self):
        X = pd.DataFrame([['foo', 2],
                          ['bar', 1],
                          ['foo', 6],
                          ['bar', 10],
                          ['foo', np.nan],
                          ['bar', np.nan]],
                         columns=['a', 'b'])
        result = (ConvertStrToInt(target_column='a', str_list=['foo', 'bar'])
                  .fit_transform(X))
        self.assertEqual(result.loc[0, 'a'], 0)
        self.assertEqual(result.loc[1, 'a'], 1)

        result = (ConvertStrToInt(target_column='a', str_list=['bar', 'foo'])
                  .fit_transform(X))
        self.assertEqual(result.loc[0, 'a'], 1)
        self.assertEqual(result.loc[1, 'a'], 0)

    def test_classifier_ensemble(self):
        model1 = lgb.LGBMClassifier(random_state=0)
        model2 = lgb.LGBMClassifier(random_state=1)
        model3 = lgb.LGBMClassifier(random_state=2)

        ensemble = ClassifierEnsemble([model1, model2, model3])
        ensemble.fit(self.cls_x_data, self.cls_y_data)
        model1.fit(self.cls_x_data, self.cls_y_data)
        model2.fit(self.cls_x_data, self.cls_y_data)
        model3.fit(self.cls_x_data, self.cls_y_data)

        ensemble_pred = ensemble.predict(self.cls_x_data)
        model1_pred = model1.predict(self.cls_x_data)
        model2_pred = model1.predict(self.cls_x_data)
        model3_pred = model1.predict(self.cls_x_data)

        mode = (stats.mode(np.column_stack([model1_pred, model2_pred,
                                            model3_pred]), axis=1)[0].T[0])
        self.assertListEqual(mode.tolist(), ensemble_pred.tolist())

    def test_classifier_ensemble_sample_weight(self):
        """ Ensemble with sample weights """
        model1 = lgb.LGBMClassifier(random_state=0)
        model2 = lgb.LGBMClassifier(random_state=1)
        model3 = lgb.LGBMClassifier(random_state=2)

        ensemble = ClassifierEnsemble([model1, model2, model3])
        ensemble.fit(self.cls_x_data, self.cls_y_data, sample_weight=self.cls_x_data[0])
        model1.fit(self.cls_x_data, self.cls_y_data, sample_weight=self.cls_x_data[0])
        model2.fit(self.cls_x_data, self.cls_y_data, sample_weight=self.cls_x_data[0])
        model3.fit(self.cls_x_data, self.cls_y_data, sample_weight=self.cls_x_data[0])

        ensemble_pred = ensemble.predict(self.cls_x_data)
        model1_pred = model1.predict(self.cls_x_data)
        model2_pred = model1.predict(self.cls_x_data)
        model3_pred = model1.predict(self.cls_x_data)

        mode = (stats.mode(np.column_stack([model1_pred, model2_pred,
                                            model3_pred]), axis=1)[0].T[0])
        self.assertListEqual(mode.tolist(), ensemble_pred.tolist())

    def test_regressor_ensemble(self):
        model1 = lgb.LGBMRegressor(random_state=0)
        model2 = lgb.LGBMRegressor(random_state=1)
        model3 = lgb.LGBMRegressor(random_state=2)

        ensemble = RegressorEnsemble([model1, model2, model3])
        ensemble.fit(self.reg_x_data, self.reg_y_data)
        model1.fit(self.reg_x_data, self.reg_y_data)
        model2.fit(self.reg_x_data, self.reg_y_data)
        model3.fit(self.reg_x_data, self.reg_y_data)

        ensemble_pred = ensemble.predict(self.reg_x_data)
        model1_pred = model1.predict(self.reg_x_data)
        model2_pred = model1.predict(self.reg_x_data)
        model3_pred = model1.predict(self.reg_x_data)

        mean = (np.mean(np.column_stack([model1_pred, model2_pred,
                                         model3_pred]), axis=1))
        self.assertListEqual(mean.tolist(), ensemble_pred.tolist())
