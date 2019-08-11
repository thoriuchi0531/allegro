import unittest

import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from ..callback import ReduceLrOnPlateau


class TestCv(unittest.TestCase):
    def setUp(self):
        reg_data = load_boston()
        self.reg_x_data = pd.DataFrame(reg_data.data)
        self.reg_y_data = reg_data.target

    def test_dynamic_lr(self):
        train_idx, valid_idx = train_test_split(range(len(self.reg_x_data)), test_size=0.1)
        train_x = self.reg_x_data.iloc[train_idx, :]
        train_y = self.reg_y_data[train_idx]
        valid_x = self.reg_x_data.iloc[valid_idx, :]
        valid_y = self.reg_y_data[valid_idx]
        callback = ReduceLrOnPlateau('rmse', factor=0.5, patience=5,
                                     min_lr=0.01, cooldown=5)

        model = lgb.LGBMRegressor(n_estimators=1000, random_state=0)
        model.fit(train_x, train_y,
                  eval_set=[(valid_x, valid_y)],
                  eval_metric='rmse',
                  verbose=1000,
                  early_stopping_rounds=100,
                  callbacks=[callback])


