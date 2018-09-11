import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from ..pipeline import (FilterXGBImportance, FilterLGBImportance,
                        ConvertNaNs, FillNa, GroupFillNa,
                        ConditionalFillNa, ConvertStrToInt)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        data = load_boston()
        self.x_data = pd.DataFrame(data.data)
        self.y_data = data.target

    def test_filter_xgb_importance_threshold(self):
        """ Filter based on threshold """
        pipeline = FilterXGBImportance(threshold='mean')
        pipeline.fit(self.x_data, self.y_data)
        filtered = pipeline.transform(self.x_data)

        importance = pipeline.estimator_.feature_importances_
        importance_mean = np.mean(importance)
        flg = importance > importance_mean
        self.assertTrue(set(self.x_data.columns[flg]), set(filtered.columns))

    def test_filter_xgb_importance_n_features(self):
        """ Filter based on the number of features """
        pipeline = FilterXGBImportance(n_features=3)
        pipeline.fit(self.x_data, self.y_data)
        filtered = pipeline.transform(self.x_data)

        importance = pipeline.estimator_.feature_importances_
        importance_rank = np.argsort(importance)
        flg = importance_rank >= len(self.x_data.columns) - pipeline.n_features
        self.assertTrue(set(self.x_data.columns[flg]), set(filtered.columns))

    def test_filter_lgb_importance_threshold(self):
        """ Filter based on threshold """
        pipeline = FilterLGBImportance(threshold='mean')
        pipeline.fit(self.x_data, self.y_data)
        filtered = pipeline.transform(self.x_data)

        importance = pipeline.estimator_.feature_importances_
        importance_mean = np.mean(importance)
        flg = importance > importance_mean
        self.assertTrue(set(self.x_data.columns[flg]), set(filtered.columns))

    def test_filter_lgb_importance_n_features(self):
        """ Filter based on the number of features """
        pipeline = FilterLGBImportance(n_features=3)
        pipeline.fit(self.x_data, self.y_data)
        filtered = pipeline.transform(self.x_data)

        importance = pipeline.estimator_.feature_importances_
        importance_rank = np.argsort(importance)
        flg = importance_rank >= len(self.x_data.columns) - pipeline.n_features
        self.assertTrue(set(self.x_data.columns[flg]), set(filtered.columns))

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

