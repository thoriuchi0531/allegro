import warnings
import numpy as np
import pandas as pd
from sklearn.base import (BaseEstimator, TransformerMixin, RegressorMixin,
                          clone)
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.feature_selection.from_model import _get_feature_importances
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
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
        self.unique_indices = None

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


class FilterXGBImportance(SelectFromModel):
    def __init__(self, threshold=None, prefit=False, norm_order=1,
                 n_features=None, **xgb_params):
        super(FilterXGBImportance, self).__init__(None, threshold,
                                                  prefit, norm_order)
        self.n_features = n_features
        self.xgb_params = xgb_params

        # validate inputs
        if self.threshold is not None and self.n_features is None:
            pass
        elif self.threshold is None and self.n_features is not None:
            if not isinstance(self.n_features, (int, float)):
                raise TypeError('n_features has to be a number. '
                                'Got n_features={}'.format(self.n_features))
        else:
            raise ValueError('Got threshold={} and n_features={}. '
                             'Either of them needs to be specified.'
                             .format(self.threshold, self.n_features))

    def _get_support_mask(self):
        if self.threshold is not None:
            return super(FilterXGBImportance, self)._get_support_mask()
        elif self.threshold is None:
            if self.prefit:
                estimator = self.estimator
            elif hasattr(self, 'estimator_'):
                estimator = self.estimator_
            else:
                raise ValueError(
                    'Either fit SelectFromModel before transform or set "prefit='   
                    'True" and pass a fitted estimator to the constructor.'
                )

            score = _get_feature_importances(estimator, self.norm_order)
            rank_index = np.argsort(score)
            return rank_index < self.n_features

        else:
            raise ValueError('Got threshold={} and n_features={}. '
                             'Either of them needs to be specified.'
                             .format(self.threshold, self.n_features))

    def fit(self, X, y=None, **fit_params):
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

        self.estimator_ = xgb.XGBRegressor(n_estimators=5000, n_jobs=-1)
        self.estimator_.set_params(**xgb_params)
        self.estimator_.fit(X, y)
        return self

    def transform(self, X):
        """ Return as a DataFrame """
        mask = self.get_support()
        filtered_columns = X.columns[mask]
        filtered = super(FilterXGBImportance, self).transform(X)
        return pd.DataFrame(filtered, index=X.index,
                            columns=filtered_columns)


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

        # convert NaN to string for potential label encoding
        X.loc[:, object_columns] = (X.loc[:, object_columns].fillna('NaN')
                                    .astype('category'))
        return X


class ConvertNaNs(Imputer):
    def __init__(self, missing_values="NaN", strategy="mean",
                 axis=0, verbose=0, copy=True, target_columns=float):
        # scikit-learn estimators should always specify their parameters
        # in the signature of their __init__ (no varargs).
        self.col_all = None
        self.cols_to_convert = None
        self.target_columns = target_columns
        super(ConvertNaNs, self).__init__(
            missing_values, strategy, axis, verbose, copy
        )

    def fit(self, X, *_):
        self.col_all = X.columns
        if self.target_columns in (float, int):
            self.cols_to_convert = X.select_dtypes(self.target_columns).columns
        elif isinstance(self.target_columns, str):
            self.cols_to_convert = [self.target_columns]
        elif isinstance(self.target_columns, list):
            self.cols_to_convert = self.target_columns
        super(ConvertNaNs, self).fit(X[self.cols_to_convert])
        return self

    def transform(self, X, *_):
        col_original = X.columns
        col_others = list(set(self.col_all) - set(self.cols_to_convert))
        transformed = super(ConvertNaNs, self).transform(X[self.cols_to_convert])
        transformed = pd.DataFrame(transformed, index=X.index,
                                   columns=self.cols_to_convert)
        result = pd.concat((
            X[col_others],
            transformed
        ), axis=1).reindex(columns=col_original)
        return result


class ConvertOneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.category_columns = None
        self.ohe_columns = None

    def fit(self, X, *_):
        self.category_columns = X.select_dtypes('category').columns
        self.ohe_columns = pd.get_dummies(X[self.category_columns]).columns
        return self

    def transform(self, X, *_):
        col_others = list(set(X.columns) - set(self.category_columns))
        transformed = pd.get_dummies(X[self.category_columns])

        if set(transformed.columns) != set(self.ohe_columns):
            in_original_not_in_new = (set(self.ohe_columns) -
                                      set(transformed.columns))
            not_in_original_in_new = (set(transformed.columns) -
                                      set(self.ohe_columns))
            warnings.warn(
                '\nPresent in the original but missing in the transformed. '
                'Filled with 0s: {}\n'
                .format(in_original_not_in_new))
            warnings.warn(
                '\nPresent in the transformed but missing in the original. '
                'Removed: {}\n'
                .format(not_in_original_in_new))

        result = pd.concat((
            X[col_others],
            transformed.reindex(columns=self.ohe_columns).fillna(0)
        ), axis=1)
        return result


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
        self.models = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X)
                                       for model in self.models])
        return np.mean(predictions, axis=1)
