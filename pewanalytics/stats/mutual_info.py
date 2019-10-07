from __future__ import division
import math
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from pewtils import is_not_null, scale_range


def compute_mutual_info(y, x, weights=None, col_names=None, l=0, normalize=True):

    """
    Computes pointwise mutual information for a set of observations partitioned into two groups. The function expects
    `y` to correspond to a list or series of values indicating which partition an observation belongs two. `y` must
    be a binary flag. `x` is a set of features (either a DataFrame or sparse matrix) where the rows correspond to
    observations and the columns represent the presence of features (you can technically run this using non-binary
    features but the results will not be as readily interpretable.) The function returns a DataFrame of metrics
    computed for each feature, including:
    - `MI1`: the feature's mutual information for the positive class
    - `MI2`: the feature's mutual information for the negative class
    - `total`: the total number of times a feature appeared
    - `total_pos_with_term`: the total number of times a feature appeared in positive cases
    - `total_neg_with_term`: the total number of times a feature appeared in negative cases
    - `total_pos_neg_with_term_diff`: the raw difference in the number of times a feature appeared in positive cases
    relative to negative cases
    - `pct_pos_with_term`: the proportion of positive cases that had the feature
    - `pct_neg_with_term`: the proportion of negative cases that had the feature
    - `pct_pos_neg_with_term_ratio`: a likelihood ratio indicating the degree to which positive case was more likely to
    have the feature than a negative case
    - `pct_term_pos`: of the cases that had a feature, the proportion that were in the positive class
    - `pct_term_neg`: of the cases that had a feature, the proportion that were in the negative class
    - `pct_term_pos_neg_diff`: the percentage point difference between the proportion of cases with the feature that
    were positive vs. negative
    - `pct_term_pos_neg_ratio`: a likelihood ratio indicating the degree to which a feature was more likely to appear
    in a positive case relative to a negative one (may not be meaningful when classes are imbalanced)

    Note that `pct_term_pos` and `pct_term_neg` may not be directly comparable if classes are imbalanced, and in such
    cases a `pct_term_pos_neg_diff` above zero or `pct_term_pos_neg_ratio` above 1 may not indicate a true association
    with the positive class if positive cases outnumber negative ones.

    :param y: An array or, preferably, a pandas.Series
    :param x: A matrix, pandas.DataFrame, or preferably a Scipy csr_matrix
    :param col_names: The feature names associated with the columns in matrix 'x'
    :param l: An optional Laplace smoothing parameter
    :param normalize: Toggle normalization on or off (to control for feature prevalance), on by default
    :return: A dataframe of features with a variety of computed metrics including mutual information.
    """

    if is_not_null(weights):
        weights = weights.fillna(0)
        y0 = sum(weights[y == 0])
        y1 = sum(weights[y == 1])
        total = sum(weights)
    else:
        y0 = len(y[y == 0])
        y1 = len(y[y == 1])
        total = y1 + y0

    if type(x).__name__ == "csr_matrix":

        if is_not_null(weights):
            x = x.transpose().multiply(csr_matrix(weights)).transpose()
        x1 = pd.Series(x.sum(axis=0).tolist()[0])
        x0 = total - x1
        x1y0 = pd.Series(
            x[np.ravel(np.array(y[y == 0].index)), :].sum(axis=0).tolist()[0]
        )
        x1y1 = pd.Series(
            x[np.ravel(np.array(y[y == 1].index)), :].sum(axis=0).tolist()[0]
        )

    else:

        if type(x).__name__ != "DataFrame":
            x = pd.DataFrame(x, columns=col_names)

        x1 = x.multiply(weights, axis="index").sum()
        x0 = ((x * -1) + 1).multiply(weights, axis="index").sum()

        if is_not_null(weights):
            x = x.multiply(weights, axis="index")
        x1y0 = x[y == 0].sum()
        x1y1 = x[y == 1].sum()

    px1y0 = x1y0 / total
    px1y1 = x1y1 / total
    px0y0 = (y0 - x1y0) / total
    px0y1 = (y1 - x1y1) / total

    px1 = x1 / total
    px0 = x0 / total
    py1 = float(y1) / float(total)
    py0 = float(y0) / float(total)

    MI1 = (px1y1 / (px1 * py1) + l).map(lambda v: math.log(v, 2) if v > 0 else 0)
    if normalize:
        MI1 = MI1 / (-1 * px1y1.map(lambda v: math.log(v, 2) if v > 0 else 0))

    MI0 = (px1y0 / (px1 * py0) + l).map(lambda v: math.log(v, 2) if v > 0 else 0)
    if normalize:
        MI0 = MI0 / (-1 * px1y0.map(lambda v: math.log(v, 2) if v > 0 else 0))

    df = pd.DataFrame()

    df["MI1"] = MI1
    df["MI0"] = MI0

    df["total"] = x1
    df["total_pos_with_term"] = x1y1  # total_pos_mention
    df["total_neg_with_term"] = x1y0  # total_neg_mention
    df["total_pos_neg_with_term_diff"] = (
        df["total_pos_with_term"] - df["total_neg_with_term"]
    )

    df["pct_pos_with_term"] = x1y1 / y1  # pct_pos_mention
    df["pct_neg_with_term"] = x1y0 / y0  # pct_neg_mention
    df["pct_pos_neg_with_term_diff"] = (
        df["pct_pos_with_term"] - df["pct_neg_with_term"]
    )  # pct_pos_neg_mention_diff
    df["pct_pos_neg_with_term_ratio"] = df["pct_pos_with_term"] / (
        df["pct_neg_with_term"]
    )  # pct_pos_neg_mention_ratio

    df["pct_term_pos"] = x1y1 / x1  # pct_mention_pos
    df["pct_term_neg"] = x1y0 / x1  # pct_mention_neg
    df["pct_term_pos_neg_diff"] = (
        df["pct_term_pos"] - df["pct_term_neg"]
    )  # pct_mention_pos_neg_diff
    df["pct_term_pos_neg_ratio"] = df["pct_term_pos"] / df["pct_term_neg"]

    if col_names:
        df.index = col_names

    return df


def mutual_info_bar_plot(
    mutual_info,
    filter_col="MI1",
    top_n=50,
    x_col="pct_term_pos_neg_ratio",
    color="grey",
    title=None,
    width=10,
):
    """
    Takes a mutual information table generated by `compute_mutual_info`, selects the top features based on `filter_col`
    and then generates a bar plot of top features sorted by `x_col`.  Allows for an easy visualization of feature
    differences. Can subsequently call `plt.show()` or `plt.savefig` to display or save the plot.

    :param mutual_info: A mutual information table generated by `compute_mutual_info`
    :param filter_col: The column to use when selecting top features; sorts in descending order and picks the top
    `top_n`
    :param top_n: The number of features to display
    :param x_col: The column by which to sort the final set of top features (after they have been selected by
    `filter_col`
    :param color: The color of the bars
    :param title: The title of the plot
    :param width: The width of the plot
    :return: A Matplotlib figure, which you can display via `plt.show()` or alternatively save to a file via `plt.savefig(FILEPATH)`
    """

    import seaborn as sns
    import matplotlib.pyplot as plt

    mutual_info = mutual_info.sort_values(filter_col, ascending=False)[:top_n]
    mutual_info = mutual_info.sort_values(x_col, ascending=False)
    mutual_info["ngram"] = mutual_info.index
    buffer = 0.02 * abs(mutual_info[x_col].max() - mutual_info[x_col].min())
    plt.figure(figsize=(width, float(len(mutual_info) * 0.35)))
    sns.set_color_codes("pastel")
    g = sns.barplot(x=x_col, y="ngram", data=mutual_info, color=color)
    sns.despine(offset=10, trim=True)
    for i, row in enumerate(mutual_info.iterrows()):
        index, row = row
        g.text(
            x=row[x_col] + buffer,
            y=i,
            s=row["ngram"],
            horizontalalignment="left",
            verticalalignment="center",
            size="large",
            color=color,
        )
    g.set_title(title)

    return g


def mutual_info_scatter_plot(
    mutual_info,
    filter_col="MI1",
    top_n=50,
    x_col="pct_term_pos_neg_ratio",
    xlabel=None,
    scale_x_even=True,
    y_col="MI1",
    ylabel=None,
    scale_y_even=True,
    color="grey",
    color_col="MI1",
    size_col="pct_pos_with_term",
    title=None,
    figsize=(10, 10),
):
    """

    Takes a mutual information table generated by `compute_mutual_info`, selects the top features based on `filter_col`
    and then generates a scatter plot of top features by `x_col` and `y_col`. The names of the features will be
    displayed with varying colors and sizes depending on the variables specified in `color_col` and `size_col`.
    Allows for an easy visualization of feature differences. Can subsequently call `plt.show()` or `plt.savefig` to
    display or save the plot.

    :param mutual_info: A mutual information table generated by `compute_mutual_info`
    :param filter_col: The column to use when selecting top features; sorts in descending order and picks the top
    `top_n`
    :param top_n: The number of features to display
    :param x_col: The column to use as the x-axis
    :param xlabel: Label for the x-axis
    :param scale_x_even: If True, set values to their ordered rank (allows for even spacing)
    :param y_col: The column to use as the y-axis
    :param ylabel: Label for the y-axis
    :param scale_y_even: If True, set values to their ordered rank (allows for even spacing)
    :param color: The color for the features
    :param color_col: The column to use when shading the features
    :param size_col: The column to use to size the features
    :param title: The title of the plot
    :param figsize: The size of the plot (tuple)
    :return: A Matplotlib figure, which you can display via `plt.show()` or alternatively save to a file via `plt.savefig(FILEPATH)`
    """

    import matplotlib.pyplot as plt

    mutual_info = mutual_info.sort_values(filter_col, ascending=False)[:top_n]

    if scale_x_even:
        mutual_info = mutual_info.sort_values(x_col)
        mutual_info["{}_rank".format(x_col)] = mutual_info.reset_index().index + 1
        x_col = "{}_rank".format(x_col)
    if scale_y_even:
        mutual_info = mutual_info.sort_values(y_col)
        mutual_info["{}_rank".format(y_col)] = mutual_info.reset_index().index + 1
        y_col = "{}_rank".format(y_col)

    color_maps = {
        "grey": plt.cm.Greys,
        "purple": plt.cm.Purples,
        "blue": plt.cm.Blues,
        "green": plt.cm.Greens,
        "orange": plt.cm.Oranges,
        "red": plt.cm.Reds,
    }

    f, ax = plt.subplots(figsize=figsize)
    mutual_info["size"] = mutual_info[size_col].map(
        lambda x: scale_range(
            x, mutual_info[size_col].min(), mutual_info[size_col].max(), 15, 25
        )
    )
    mutual_info["color"] = mutual_info[color_col].map(
        lambda x: scale_range(
            x, mutual_info[color_col].min(), mutual_info[color_col].max(), 0.4, 1
        )
    )
    mutual_info["color"] = mutual_info["color"].map(color_maps[color])

    mutual_info["x"] = mutual_info[x_col]
    mutual_info["y"] = mutual_info[y_col]
    ax.set_title(title)
    ax.set_xlim((mutual_info["x"].min() * 0.9, mutual_info["x"].max() * 1.1))
    ax.set_ylim((mutual_info["y"].min() * 0.9, mutual_info["y"].max() * 1.1))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    texts = []
    for index, row in mutual_info.iterrows():
        texts.append(
            ax.text(row["x"], row["y"], index, size=row["size"], color=row["color"])
        )

    return ax
