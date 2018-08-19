import warnings
import numpy as np
import pandas as pd
from sklearn.base import (BaseEstimator, TransformerMixin, RegressorMixin,
                          clone)
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import FeatureUnion
import xgboost as xgb


# ------------------------------------------------------------------------------
# filter
# ------------------------------------------------------------------------------
def _transform_one(transformer, weight, X):
    res = transformer.transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


class VarianceThresholdDF(VarianceThreshold):
    def transform(self, X, *_):
        # return pd.DataFrame
        used_cols = [col for col, flg in zip(X.columns,
                                             self._get_support_mask()) if flg]
        return X[used_cols]


class FilterUnique(BaseEstimator, TransformerMixin):
    def __init__(self, axis=1):
        self.axis = axis

    def fit(self, X, *_):
        _, self.unique_indices = np.unique(X, axis=self.axis, return_index=True)
        return self

    def transform(self, X, *_):
        if self.axis == 1:
            return X.iloc[:, self.unique_indices]
        elif self.axis == 0:
            return X.iloc[self.unique_indices, :]
        else:
            raise NotImplementedError()


class FilterXGBImportance(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, **xgb_params):
        self.n_features = n_features
        self.model = None
        self.xgb_params = xgb_params
        self.xgb_f_score = None

    def fit(self, X, y):
        model_tmp = xgb.XGBRegressor(n_estimators=5000, n_jobs=-1)
        model_tmp.set_params(**self.xgb_params)

        xgb_dmatrix = xgb.DMatrix(X, y)
        xgb_params = model_tmp.get_xgb_params()
        xgb_cv_result = xgb.cv(
            xgb_params,
            xgb_dmatrix,
            num_boost_round=xgb_params.get('n_estimators'),
            nfold=10,
            metrics='rmse',
            early_stopping_rounds=10,
            verbose_eval=False
        )
        n_estimators = xgb_cv_result.shape[0]
        xgb_params['n_estimators'] = n_estimators

        self.model = xgb.XGBRegressor(n_estimators=5000, n_jobs=-1)
        self.model.set_params(**xgb_params)
        self.model.fit(X, y)

        self.xgb_f_score = (pd.Series(self.model.feature_importances_,
                                      index=X.columns)
                            .sort_values(ascending=False))
        return self

    def transform(self, X, *_):
        return X.loc[:, self.xgb_f_score.head(self.n_features).index]


# ------------------------------------------------------------------------------
# converter
# ------------------------------------------------------------------------------
class ConvertToLog1p(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self

    def transform(self, X, *_):
        return np.sign(X) * np.log1p(np.abs(X))


class ConvertToCategory(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self

    def transform(self, X, *_):
        X = X.copy(True)
        object_columns = X.select_dtypes(object).columns
        n_category = X.loc[:, object_columns].nunique()

        warning_flgs = n_category > (X.shape[0] * 0.5)
        if warning_flgs.any():
            warnings.warn('Some columns contain too many categories: {}'
                          .format(n_category[warning_flgs].index))

        X.loc[:, object_columns] = X.loc[:, object_columns].astype('category')
        return X


class ConvertNaNs(BaseEstimator, TransformerMixin):
    def __init__(self, **float_config):
        self.float_median = float_config.pop('median', [])
        self.float_mode = float_config.pop('mode', [])
        self.float_pad = float_config.pop('pad', [])

        self.float_median = self._to_list(self.float_median)
        self.float_mode = self._to_list(self.float_mode)
        self.float_pad = self._to_list(self.float_pad)

    @staticmethod
    def _to_list(value):
        if isinstance(value, list):
            return value
        else:
            return [value]

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        """ Transform missing values

        float columns
            fill with median by default

        object columns
            do nothing at the moment
        """
        X = X.copy(True)
        na_columns = X.columns[X.isna().any()]
        na_float = X.loc[:, na_columns].select_dtypes(float).columns
        na_object = X.loc[:, na_columns].select_dtypes(object).columns
        na_category = X.loc[:, na_columns].select_dtypes('category').columns
        na_remaining = (set(na_columns) - set(na_float) - set(na_object) -
                        set(na_category))

        if len(na_remaining) != 0:
            raise ValueError('Unrecognised na columns: {}'.format(na_remaining))

        # float columns
        for i in na_float:
            if i in self.float_mode:
                X[i] = X[i].fillna(X[i].mode())
            elif i in self.float_pad:
                X[i] = X[i].fillna(method='pad')
            else:
                X[i] = X[i].fillna(X[i].median())

        return X


# ------------------------------------------------------------------------------
# features
# ------------------------------------------------------------------------------
class FeatureUnionDF(FeatureUnion):
    def transform(self, X, y=None):
        Xs = Parallel(n_jobs=self.n_jobs)(delayed(_transform_one)(trans, weight, X)
                                          for name, trans, weight in self._iter())
        Xs = pd.concat(Xs, axis=1)
        return Xs


class FeatureRaw(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self

    def transform(self, X, *_):
        return X


class FeatureTransformed(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, name=None):
        self.estimator = estimator
        self.name = name or self.estimator.__class__.__name__

    def fit(self, X, *_):
        self.estimator.fit(X)
        return self

    def transform(self, X, *_):
        transformed = self.estimator.transform(X)
        col_names = ['{}_{}'.format(self.name, i + 1)
                     for i in range(transformed.shape[1])]
        df = pd.DataFrame(transformed, index=X.index, columns=col_names)
        return df


class FeatureAggregated(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self

    def transform(self, X, *_):
        df = pd.DataFrame()
        x_non_zero = X.where(lambda x: x != 0)
        return (df
            .assign(
            count_zeros=X.sum(axis=1),
            sum_values=X.sum(axis=1),
            mean=X.mean(axis=1),
            median=X.median(axis=1),
            var=X.var(axis=1),
            skew=X.skew(axis=1),
            kurt=X.kurt(axis=1),
            max_values=X.max(axis=1),
            nunique=X.nunique(axis=1),

            count_zeros_non_zero=x_non_zero.sum(axis=1),
            sum_values_non_zero=x_non_zero.sum(axis=1),
            mean_non_zero=x_non_zero.mean(axis=1),
            median_non_zero=x_non_zero.median(axis=1),
            var_non_zero=x_non_zero.var(axis=1),
            skew_non_zero=x_non_zero.skew(axis=1),
            kurt_non_zero=x_non_zero.kurt(axis=1),
            max_values_non_zero=x_non_zero.max(axis=1),
            nunique_non_zero=x_non_zero.nunique(axis=1),
        ))


# ------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------
class ModelEnsemble(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X)
                                       for model in self.models_])
        return np.mean(predictions, axis=1)
