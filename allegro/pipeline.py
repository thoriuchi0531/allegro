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


class UniqueTransformer(BaseEstimator, TransformerMixin):
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


class XGBImportanceFilter(BaseEstimator, TransformerMixin):
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
class LnTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self

    def transform(self, X, *_):
        return np.sign(X) * np.log1p(np.abs(X))


# ------------------------------------------------------------------------------
# features
# ------------------------------------------------------------------------------
class FeatureUnionDF(FeatureUnion):
    def transform(self, X, y=None):
        Xs = Parallel(n_jobs=self.n_jobs)(delayed(_transform_one)(trans, weight, X)
                                          for name, trans, weight in self._iter())
        Xs = pd.concat(Xs, axis=1)
        return Xs


class DoNothing(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self

    def transform(self, X, *_):
        return X


class TransformedFeatures(BaseEstimator, TransformerMixin):
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


class AggFeatures(BaseEstimator, TransformerMixin):
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
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
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
