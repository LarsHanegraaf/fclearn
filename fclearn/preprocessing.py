"""Scikit-learn style transformer that can be used for time series problems."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import OneHotEncoder

from fclearn.pandas_helpers import get_time_series_combinations


def _occurrence_value(x, order):
    # result = collections.Counter(array).most_common(order)[order - 1][0]
    try:
        result = x.fillna(0).value_counts().index.values[order - 1]
    except:  # noqa: E722
        result = 0
    return result


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


class LastWeekZeroTransformer(BaseEstimator, TransformerMixin):
    """Checks whether last week was zero.

    Creats a column with a boolean whether last week was zero or not.
    """

    def fit(self, X, y=None):
        """Fit the transformer to X."""
        return self

    def transform(self, X):
        """Transform X."""
        assert isinstance(X, pd.DataFrame), "X should be of type dataframe"
        X_ = X[["t-1"]]
        X_ = X_.rename(columns={"t-1": "last_week_zero"})
        X_["last_week_zero"] = X_["last_week_zero"].apply(lambda x: 1 if x == 0 else 0)
        return X_


class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
    """Wrapper for transformers that should be applied to indiviual series.

    Transformer that applies the transformer passed to the argument to every individual
    time serie.
    The dataset this transformer is fit to, should already be ordere using groupby and
    apply
    """

    def __init__(self, transformer, groupby, num_series=None):
        """Constructor."""
        self.transformer = transformer
        self.groupby = groupby
        self.num_series = num_series

    def fit(self, X, y=None):
        """Fit the transformer to the series."""
        # Check type
        if not isinstance(X, pd.DataFrame):
            if self.num_series is None:
                raise ValueError(
                    "num_series has to be set if X is not of type DataFrame"
                )
        else:
            # Calculate the size of each individual serie in the dataframe. Used for
            # grouping the scalers
            self.num_series = len(get_time_series_combinations(X, self.groupby))

        serie_size = X.shape[0] / self.num_series

        if not serie_size.is_integer():
            raise ValueError("Size of individual series in X is not equal")

        serie_size = int(serie_size)

        # For each individual serie, fit a scaler and store the fitted results into
        # self.transformers
        self.transfomers = []

        for serie_index in range(self.num_series):
            start = 0 + serie_index * serie_size
            stop = serie_size + serie_index * serie_size

            transformer = clone(self.transformer)

            if y is not None:

                self.transfomers.append(transformer.fit(X[start:stop], y[start:stop]))

            else:
                self.transfomers.append(transformer.fit(X[start:stop]))

        return self

    def transform(self, X):
        """Transform the series X."""
        # Check whether input data has a shape that is a multiple of the amount of
        # series
        serie_size = X.shape[0] / self.num_series

        if not serie_size.is_integer():
            raise ValueError("Size of individual series in X is not equal")

        serie_size = int(serie_size)

        X_ = X.copy()

        Xs = []

        # Transform every time serie individually
        for serie_index in range(self.num_series):
            start = 0 + serie_index * serie_size
            stop = serie_size + serie_index * serie_size

            result = self.transfomers[serie_index].transform(X_[start:stop])

            X_[start:stop] = result
            # validate next two lines
            Xs.append(result)

        Xs = pd.concat(Xs)
        return Xs
        # return X_

    def inverse_transform(self, X):
        """Transform X back to it's original distribution."""
        # Check whether input data has a shape that is a multiple of the amount of
        # series
        serie_size = X.shape[0] / self.num_series

        if not serie_size.is_integer():
            raise ValueError("Size of individual series in X is not equal")

        serie_size = int(serie_size)

        X_ = X.copy()

        # Transform every time serie individually
        for serie_index in range(self.num_series):
            start = 0 + serie_index * serie_size
            stop = serie_size + serie_index * serie_size

            X_[start:stop] = self.transfomers[serie_index].inverse_transform(
                X_[start:stop]
            )

        return X_


class RowSelector(BaseEstimator, TransformerMixin):
    """Select rows based on a criterion.

    The criterion is matched to be equal.

    Args:
        column (string): Column to do the look up information on.
        expression (Any): Value to match the row on in the column.
        drop_column_after_selection (bool): Whether the column that has been searched
            should be deleted after filtering.
    """

    def __init__(self, column, expression, drop_column_after_selection=True):
        """Constructor."""
        self.column = column
        self.expression = expression
        self.drop_column_after_selection = drop_column_after_selection

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
        assert isinstance(X, pd.DataFrame), "X should be of type DataFrame"
        X_ = X.copy()
        X_ = X_.loc[X_[self.column] == self.expression]
        if self.drop_column_after_selection:
            X_ = X_.drop(columns=[self.column])
        return X_


class OccurrenceTransformer(BaseEstimator, TransformerMixin):
    """Transformer that creates a feature with the most occurring target frequency.

    Preferably used in combinaten with the TimeSeriesTransformer

    Args:
        order (int): Order to create a feature for. 1 is most occurring target value
            2 is second most occurring target.
    """

    def __init__(self, order=1):
        """Constructor."""
        self.order = order

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
        """Transform DataFrame X by creating a column with target occurrence.

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: **X_new** - DataFrame with occurrence of the target.
        """
        assert isinstance(X, pd.DataFrame), "X should be of type DataFrame"
        assert self.order >= 1, "order should be greater or equal to 1"
        X_ = X.copy()
        column_name = "occurrence-{}".format(self.order)
        X_[column_name] = X_.apply(
            lambda x: _occurrence_value(x, self.order),
            axis=1,
            raw=False,
        )
        X_ = X_[[column_name]]

        return X_


class MovingAggregateTransformer(BaseEstimator, TransformerMixin):
    """Transformer for aggregating multiple columns of the same row.

    Note that this behaviour is different from :class:`RollingAggregateTransformer`
    where this class aggregates over columns, wheareas the
    RollingAggregateTransformer aggregates within the column.
    """

    def __init__(self, window, aggregate):
        """Constructor."""
        self.window = window
        self.aggregate = aggregate

    def fit(self, X, y=None):
        """Fit transformer to the data."""
        return self

    def transform(self, X):
        """Tranfrom the data."""
        assert isinstance(X, pd.DataFrame), "X should be of type DataFrame"
        X_ = X.iloc[:, -self.window :].agg(func=self.aggregate, axis=1)
        X_.name = "t-{}-{}".format(self.aggregate, self.window)
        return X_


class RollingAggregateTransformer(BaseEstimator, TransformerMixin):
    """Transformer that aggregates within columns.

    Calculate the aggregate of days in the window. Start date is a the beginning of the
    window window = 7 and aggregation = max return the maximum value of a column for the
    next 7 days Can use a dict for manually setting different aggregation methods for
    different columns.
    """

    def __init__(self, groupby, window=7, aggregation="max"):
        """Constructor."""
        self.groupby = groupby
        self.window = window
        self.aggregation = aggregation

    def fit(self, X, y=None):
        """Fit transformer to the data."""
        return self

    def transform(self, X):
        """Transform X."""
        assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
        X_ = X.groupby(self.groupby).apply(
            lambda x: x.rolling(self.window)
            .agg(self.aggregation)
            .shift(-self.window + 1)
        )
        # Remove extra index levels caused by groupby
        # X_.index = X_.index.droplevel([0, 1])
        X_ = X_.add_suffix("_agg")
        return X_


class FillNaTransformer(BaseEstimator, TransformerMixin):
    """Transformer that fills missing values."""

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def transform(self, X):
        """Transform the data."""
        X_ = X.copy()
        for column in X_:
            if (
                X_[column].dtype.type == np.object_
                or X_[column].dtype.type == "category"
            ):
                X_[column] = X_[column].astype("str")
                X_[column] = X_[column].fillna("Unknown")
            else:
                X_[column] = X_[column].fillna(0)
        return X_


class GroupSelector(BaseEstimator, TransformerMixin):
    """Transformer that selects groups of columns, based on a string.

    This transformer selects groups based on wheter their columns name
    mathces the start of the strings.
    """

    def __init__(self, groups):
        """Constructor."""
        self.groups = groups

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def transform(self, X):
        """Transform X."""
        assert isinstance(X, pd.DataFrame)
        if self.groups is None:
            return pd.DataFrame(index=X.index)

        regex = ""

        if len(self.groups) == 0:
            return X
        else:
            for index in range(len(self.groups)):
                if index == 0:
                    regex += "^{}".format(self.groups[index])
                else:
                    regex += "|^{}".format(self.groups[index])

            return X.filter(regex=regex)


class ShiftTransformer(BaseEstimator, TransformerMixin):
    """Shifts a DataFrame for an x amount of weeks."""

    def __init__(self, shifts=[0]):
        """Constructor."""
        self.shifts = shifts

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def transform(self, X):
        """Transform X."""
        X_ = X.copy()
        Xs = []
        for shift in self.shifts:
            if shift == 0:
                shift_name = " t"
            else:
                shift_name = (
                    " t-{}".format(np.abs(shift))
                    if shift < 0
                    else " t+{}".format(shift)
                )
            X_ = X.copy()
            X_ = X_.shift(
                -shift * 7
            )  # e.g. next week carnaval (t + 1, thus index 1), needs shift -7
            X_ = X_.add_suffix(shift_name)
            Xs.append(X_)

        Xs = pd.concat(Xs, axis=1)

        return Xs


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


class PandasOneHotEncoder(BaseEstimator, TransformerMixin):
    """Sklearn One Hot encoder that supports column names.

    Should be used in combination with :class:`ToCategoryTransformer`.
    """

    def fit(self, X, y=None):
        """Fit the transformer on the data X.

        Args:
            X (pd.DataFrame): DataFrame to fit transformer on
            y (Any): y

        Returns:
            object: **self** - Estimator instance.
        """
        X_ = X.copy()
        self.encoder = OneHotEncoder(sparse=False)
        self.encoder.fit(X_)
        return self

    def transform(self, X):
        """Transform DataFrame X.

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: **X_new** - DataFrame with the one hot encoded columns.
        """
        X_ = X.copy()
        X_ = self.encoder.transform(X_)
        features = X.columns
        columns = []
        for index in range(len(self.encoder.categories_)):
            feature = features[index]
            for value in self.encoder.categories_[index]:
                columns.append("{}_{}".format(feature, value))

        df = pd.DataFrame(index=X.index, data=X_, columns=columns)
        return df


class ToCategoryTransformer(BaseEstimator, TransformerMixin):
    """Transformer that makes all the columns of datatype 'category'."""

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
        """Transform DataFrame X.

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: **X_new** - DataFrame with the category columns.
        """
        return X.astype("category")
