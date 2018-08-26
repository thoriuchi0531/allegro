import unittest

import pandas as pd
from sklearn.datasets import load_iris
from ..cv import run_xgb_cv


class TestCv(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        self.x_data = pd.DataFrame(data.data)
        self.y_data = data.target

    def test_run_xgb_cv_seed(self):
        seed0 = run_xgb_cv(self.x_data, self.y_data)
        self.assertEqual(seed0.get_xgb_params()['seed'], 0)

        seed1 = run_xgb_cv(self.x_data, self.y_data, random_state=1)
        self.assertEqual(seed1.get_xgb_params()['seed'], 1)
