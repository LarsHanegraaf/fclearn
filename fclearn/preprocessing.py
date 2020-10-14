"""Scikit-learn style transformer that can be used for time series problems."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _create_regression_task(df, target_column, number_of_lags):
    """Creates a regression task for one time series.

    The output is a dataframe with four target variables, y1, y2, y3, y4 and the lags

    y1 = amount of hectolitres for day 1 to 7 (where day 1, the day of the index
        itself is)
    y2 = amount of hectolitres for day 8 to 14
    y3 = amount of hectolitres for day 15 to 21
    y4 = amount of hectolitres for day 22 to 28

    t-1 is the amount summed of the target for the previous seven days.

    Args:
        df (pd.DataFrame): DataFrame with one time series with daily data.
        target_column (string): Name of the target column
        number_of_lags (int): Number of lags to generate

    Returns:
        pd.DataFrame: **df_new** - DataFram in which the time series is formulated as
        regression task

    """
    # Standard code
    df = df.copy()
    df = df.reset_index()
    df = df[[target_column, "Date"]]
    df.set_index("Date", inplace=True)

    #
    df["weekly_grouped"] = df[[target_column]].rolling(7).sum()

    # Create lagged columns
    for index in range(number_of_lags):
        df["t-{}".format(index + 1)] = df["weekly_grouped"].shift(1 + 7 * (index))

    # Create target columns
    df["y1"] = df["weekly_grouped"].shift(-6)
    df["y2"] = df["weekly_grouped"].shift(-6 - 7)
    df["y3"] = df["weekly_grouped"].shift(-6 - 14)
    df["y4"] = df["weekly_grouped"].shift(-6 - 21)

    df.drop(columns=["weekly_grouped", "HL_sum"], inplace=True)
    return df


class DateSelector(BaseEstimator, TransformerMixin):
    """Selects rows based on date criteria.

    Attributes:
        start (string): Rows that should be included after this date
        stop (string): Rows that should be included up untill this date
        weekday (int): Weekday that should be included (0 is Monday)
    """

    def __init__(self, start=None, stop=None, weekday=None):
        """Constructor."""
        self.start = start
        self.stop = stop
        self.weekday = weekday

    def fit(self, X, y=None):
        """Fit the transformer on the data X.

        Args:
            X (pd.DataFrame): DataFrame to fit transformer on
            y (Any): y

        Returns:
            object: **self** - Estimator instance.
        """
        return self

    def transform(self, X):
        """Transform DataFrame X. by selecting the relevant rows.

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: **X_new** - DataFrame with the selected rows.
        """
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        if self.start is not None:
            X_ = X_.loc[(X_.index.get_level_values("Date") >= self.start)]
        if self.stop is not None:
            X_ = X_.loc[(X_.index.get_level_values("Date") <= self.stop)]
        if self.weekday is not None:
            X_ = X_.loc[X_.index.get_level_values("Date").weekday == 0]
        return X_


class RegexSelector(BaseEstimator, TransformerMixin):
    """Selects columns based on a RegEx expression of their name.

    Args:
        regex (string): RegEx expression to match on
    """

    def __init__(self, regex):
        """Constructor."""
        self.regex = regex

    def fit(self, X, y=None):
        """Fit the transformer on the data X.

        Args:
            X (pd.DataFrame): DataFrame to fit transformer on
            y (Any): y

        Returns:
            object: **self** - Estimator instance.
        """
        return self

    def transform(self, X):
        """Transform DataFrame X. by selecting the columns.

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: **X_new** - DataFrame with the selected columns.
        """
        assert isinstance(X, pd.DataFrame)
        return X.filter(regex=self.regex)


class LagsSortTransformer(BaseEstimator, TransformerMixin):
    """Transformer that sorts columns created by a :class:`RegressionTransformer`."""

    def fit(self, X, y=None):
        """Fit the transformer on the data X.

        Args:
            X (pd.DataFrame): DataFrame to fit transformer on
            y (Any): y

        Returns:
            object: **self** - Estimator instance.
        """
        return self

    def transform(self, X):
        """Transform DataFrame X and return sorted lags.

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: **X_new** - DataFrame with sorted columns
        """
        assert isinstance(X, pd.DataFrame)
        columns = list(X.columns)
        columns.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
        return X[columns]


class DropNaTransformer(BaseEstimator, TransformerMixin):
    """Transformer that drops empty rows.

    Attributes:
        subset (list): List of columns to look for NA values.
    """

    def __init__(self, subset):
        """Constructor."""
        self.subset = subset

    def fit(self, X, y=None):
        """Fit the transformer on the data X.

        Args:
            X (pd.DataFrame): DataFrame to fit transformer on
            y (Any): y

        Returns:
            object: **self** - Estimator instance.
        """
        return self

    def transform(self, X):
        """Transform DataFrame X and return withou NA values.

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: **X_new** - DataFrame without empty values.
        """
        assert isinstance(X, pd.DataFrame), "X should be of type DataFrame"
        assert isinstance(self.subset, list), "subset should be of type list"
        return X.dropna(subset=self.subset)


class RegressionTransformer(BaseEstimator, TransformerMixin):
    """Transforms a time series to a regression task.

    This transformer generates for every point in time a regression 'task',
    where the target is the value of the next 7 days and the features are the
    sums of the target for the last 'lag' amount of weeks

    Args:
        groupby (list): List that is used to group a DFU on
        lags (int): Number of langs that should be created for the regression task.
        target_name (string): Target column that should be used to create a regression
            task.
    """

    def __init__(self, groupby, lags=12, target_name="y"):
        """Constructor."""
        self.groupby = groupby
        self.lags = lags
        self.target_name = target_name

    def fit(self, X, y=None):
        """Fit the transformer on the data X.

        Args:
            X (pd.DataFrame): DataFrame to fit transformer on
            y (Any): y

        Returns:
            object: **self** - Estimator instance.
        """
        return self

    def transform(self, X):
        """Transform DataFrame X and return a regression task.

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: **X_new** - DataFrame with target and the lags.
        """
        return (
            X.groupby(self.groupby)
            .apply(lambda x: _create_regression_task(x, "HL_sum", self.lags))
            .round(3)
        )


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a column from a Pandas DataFrame.

    Args:
        columns (list): List with column names to filter on
    """

    def __init__(self, columns):
        """Constructor."""
        self.columns = columns

    def fit(self, X, y=None):
        """Fit the transformer on the data X.

        Args:
            X (pd.DataFrame): DataFrame to fit transformer on
            y (Any): y

        Returns:
            object: **self** - Estimator instance.
        """
        return self

    def transform(self, X):
        """Transform DataFrame X. by selecting the columns.

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: **X_new** - DataFrame with the selected columns.
        """
        assert isinstance(X, pd.DataFrame)
        assert isinstance(self.columns, list)
        if len(self.columns) == 0:
            return X
        else:
            return X[self.columns]
