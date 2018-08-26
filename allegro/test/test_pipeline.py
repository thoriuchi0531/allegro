import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from ..pipeline import FilterXGBImportance


class TestPipeline(unittest.TestCase):
    def setUp(self):
        data = load_iris()
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
