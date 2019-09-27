from __future__ import division
import math
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from pewtils import is_not_null, scale_range


def compute_mutual_info(y, x, weights=None, col_names=None, l=0, normalize=True):
    """
    :param y: An array or, preferably, a pandas.Series
    :param x: A matrix, pandas.DataFrame, or preferably a Scipy csr_matrix
    :param col_names: The feature names associated with the columns in matrix 'x'
    :param l: An optional Laplace smoothing parameter
    :param normalize: Toggle normalization on or off (to control for feature prevalance), on by default
    :return:
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

    :param mutual_info: A mutual information table generated by `compute_mutual_info`
    :param filter_col:
    :param top_n:
    :param x_col:
    :param color:
    :param title:
    :param width:
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

    :param mutual_info: A mutual information table generated by `compute_mutual_info`
    :param filter_col:
    :param top_n:
    :param x_col:
    :param xlabel:
    :param scale_x_even:
    :param y_col:
    :param ylabel:
    :param scale_y_even:
    :param color:
    :param color_col:
    :param size_col:
    :param title:
    :param figsize:
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
    # plt.savefig(value + '.pdf')
