import unittest

import pandas as pd
from sklearn.datasets import load_boston
from ..cv import run_xgb_cv


class TestCv(unittest.TestCase):
    def setUp(self):
        data = load_boston()
        self.x_data = pd.DataFrame(data.data)
        self.y_data = data.target

    def test_run_xgb_cv_seed(self):
        seed0 = run_xgb_cv(self.x_data, self.y_data)
        self.assertEqual(seed0.get_xgb_params()['seed'], 0)

        seed1 = run_xgb_cv(self.x_data, self.y_data, random_state=1)
        self.assertEqual(seed1.get_xgb_params()['seed'], 1)


if __name__ == '__main__':
    import pandas as pd
    from sklearn.datasets import load_boston
    from allegro.cv import run_xgb_cv, run_lgb_cv
    data = load_boston()
    x_data = pd.DataFrame(data.data)
    y_data = data.target

    model = run_xgb_cv(x_data, y_data, plot_result=False)
    print(model.n_estimators)  # 142
    model = run_lgb_cv(x_data, y_data, plot_result=False)
    print(model.n_estimators)  # 132
