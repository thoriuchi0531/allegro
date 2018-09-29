import pandas as pd
from sklearn.model_selection import cross_val_score, KFold

from .log import get_logger

logger = get_logger()


def multi_kfold_cross_val_score(estimator, X, y, scoring,
                                n_splits=3, shuffle=False, n_cv=1):
    result = []
    for i in range(n_cv):
        result.append(
            cross_val_score(estimator, X, y, scoring=scoring,
                            cv=KFold(n_splits, shuffle=shuffle, random_state=i))
        )

    result = pd.DataFrame(result,
                          index=['random_state_{}'.format(i) for i in
                                 range(n_cv)],
                          columns=['split_{}'.format(i) for i in
                                   range(n_splits)])

    # plot
    average = result.mean(axis=1)
    average = average.append(pd.Series(average.mean(), index=['average']))
    logger.info('Average score: {}'.format(average.mean()))
    average.to_frame().plot(kind='bar', yerr=result.std(axis=1))
    return result
