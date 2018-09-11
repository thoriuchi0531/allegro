import warnings
import numpy as np
import pandas as pd
from sklearn.base import (BaseEstimator, TransformerMixin, RegressorMixin,
                          clone)
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.feature_selection.from_model import _get_feature_importances
from sklearn.model_selection import KFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer

from .cv import run_xgb_cv, run_lgb_cv


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


class _FilterImportance(SelectFromModel):
    def __init__(self, threshold=None, prefit=False, norm_order=1,
                 n_features=None, **params):
        super(_FilterImportance, self).__init__(None, threshold,
                                                prefit, norm_order)
        self.n_features = n_features
        self.params = params

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
            return super(_FilterImportance, self)._get_support_mask()
        elif self.n_features is not None:
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
            n_largest = np.argsort(score)[-self.n_features:]
            mask = np.zeros(len(score), dtype=bool)
            mask[n_largest] = True
            return mask

        else:
            raise ValueError('Got threshold={} and n_features={}. '
                             'Either of them needs to be specified.'
                             .format(self.threshold, self.n_features))

    def transform(self, X):
        """ Return as a DataFrame """
        mask = self.get_support()
        filtered_columns = X.columns[mask]
        filtered = super(_FilterImportance, self).transform(X)
        return pd.DataFrame(filtered, index=X.index,
                            columns=filtered_columns)


class FilterXGBImportance(_FilterImportance):
    def fit(self, X, y=None, **fit_params):
        self.estimator_ = run_xgb_cv(X, y, plot_result=False)
        return self


class FilterLGBImportance(_FilterImportance):
    def fit(self, X, y=None, **fit_params):
        self.estimator_ = run_lgb_cv(X, y, plot_result=False)
        return self


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


class ConvertStrToInt(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, str_list):
        self.target_column = target_column
        self.str_map = {i: j for i, j in zip(str_list, range(len(str_list)))}

    def fit(self, *_):
        return self

    def transform(self, X, *_):
        X = X.copy(True)
        X[self.target_column] = (X[self.target_column]
                                 .apply(lambda x: self.str_map[x]))
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


class FillNa(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, target_columns):
        self.strategy = strategy
        self.target_columns = target_columns

        if not isinstance(self.strategy, list):
            self.strategy = [self.strategy]
        if not isinstance(self.target_columns, list):
            self.target_columns = [self.target_columns]

        if len(self.strategy) != len(self.target_columns):
            raise ValueError('Length mismatch. strategy length = {} while'
                             'target column length = {}'
                             .format(len(self.strategy),
                                     len(self.target_columns)))

    def fit(self, X, *_):
        return self

    def transform(self, X, *_):
        X = X.copy(True)
        for strat, col in zip(self.strategy, self.target_columns):
            X[col] = X[col].fillna(strat)
        return X


class GroupFillNa(FillNa):
    def __init__(self, strategy, target_columns):
        super(GroupFillNa, self).__init__(strategy, target_columns)

    def transform(self, X, *_):
        X = X.copy(True)
        index_flg = X[self.target_columns].isna().all(axis=1)
        X.loc[index_flg, self.target_columns] = self.strategy
        return X


class ConditionalFillNa(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, cond_column, how='median'):
        self.target_column = target_column
        self.cond_column = cond_column
        self.how = how
        self.fill_map = None

    def fit(self, X, *_):
        fill_map = X.groupby(self.cond_column)[self.target_column]
        if self.how == 'median':
            fill_map = fill_map.median()
        elif self.how == 'mean':
            fill_map = fill_map.mean()
        else:
            raise NotImplementedError()
        self.fill_map = fill_map
        return self

    def transform(self, X, *_):
        X = X.copy(True)
        result = (X.groupby(self.cond_column)[self.target_column]
                  .transform(lambda x: x.fillna(self.fill_map[x.name])))
        X[self.target_column] = result
        return X


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
# cf) https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
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


class ModelStacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5, random_state=0):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, y):
        self.base_models_ = []
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True,
                      random_state=self.random_state)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            fold_models = []
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                fold_models.append(instance)
            self.base_models_.append(fold_models)

        # Now train the cloned  meta-model using the out-of-fold predictions
        # as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use
    # the averaged predictions as meta-features for the final prediction
    # which is done by the meta-model
    def predict(self, X):
        results = []
        for fold_models in self.base_models_:
            results.append(
                np.column_stack([model.predict(X)
                                 for model in fold_models]).mean(axis=1)
            )
        meta_features = np.column_stack(results)
        return self.meta_model_.predict(meta_features)
