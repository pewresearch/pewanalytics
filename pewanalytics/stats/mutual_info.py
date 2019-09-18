from __future__ import division
import math
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from pewtils import is_not_null, scale_range


def compute_mutual_info(
        y,
        x,
        weights=None,
        col_names=None,
        l=0,
        normalize=True,
        sort_by="MI",
        top_n=40,
        return_raw=False
):
    """
    :param y: An array or, preferably, a pandas.Series
    :param x: A matrix, pandas.DataFrame, or preferably a Scipy csr_matrix
    :param col_names: The feature names associated with the columns in matrix 'x'
    :param l: An optional Laplace smoothing parameter
    :param normalize: Toggle normalization on or off (to control for feature prevalance), on by default
    :param top_n: The number of features for each partition you want to return
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
        x1y0 = pd.Series(x[np.ravel(np.array(y[y == 0].index)), :].sum(axis=0).tolist()[0])
        x1y1 = pd.Series(x[np.ravel(np.array(y[y == 1].index)), :].sum(axis=0).tolist()[0])

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

    df['MI1'] = MI1
    df['MI0'] = MI0

    df['total'] = x1
    df['pos_freq'] = x1y1
    df['neg_freq'] = x1y0
    df['freq_diff'] = x1y1 - x1y0
    df['pct_pos_with_term'] = x1y1 / y1
    df['pct_neg_with_term'] = x1y0 / y0
    df['pct_term_pos'] = x1y1 / x1
    df['pct_term_neg'] = x1y0 / x1
    df['pct_term_diff'] = df['pct_term_pos'] - df['pct_term_neg']
    df['pct_pos_neg_diff'] = df['pct_pos_with_term'] - df['pct_neg_with_term']
    df['pct_pos_multiplier'] = df['pct_pos_with_term'] / (df['pct_neg_with_term'])
    df['pct_neg_multiplier'] = df['pct_neg_with_term'] / (df['pct_pos_with_term'])

    if col_names:
        df.index = col_names

    if return_raw:
        return df

    df.ix[df["pct_pos_neg_diff"] < 0, "MI"] = df[df["pct_pos_neg_diff"] < 0]["MI0"]
    df.ix[df["pct_pos_neg_diff"] > 0, "MI"] = df[df["pct_pos_neg_diff"] > 0]["MI1"]

    neg_df = df[df["pct_pos_neg_diff"] < 0]
    neg_df = neg_df.sort_values(sort_by, ascending=False).head(top_n)
    neg_df["freq"] = neg_df["neg_freq"]
    neg_df["pct_with_term"] = neg_df["pct_neg_with_term"]
    neg_df["pct_term"] = neg_df["pct_term_neg"]
    neg_df = neg_df.dropna(subset=[sort_by])

    pos_df = df[df["pct_pos_neg_diff"] > 0]
    pos_df = pos_df.sort_values(sort_by, ascending=False).head(top_n)
    pos_df["freq"] = pos_df["pos_freq"]
    pos_df["pct_with_term"] = pos_df["pct_pos_with_term"]
    pos_df["pct_term"] = pos_df["pct_term_pos"]
    pos_df = pos_df.dropna(subset=[sort_by])

    return (pos_df, neg_df)


def plot_mutual_info(
    mutual_info,
    color="grey",
    color_col="MI1",
    x_col="pct_pos_multiplier_rank",
    y_col="MI_rank",
    size_col="pct_pos_with_term",
    title=None,
    figsize=(10, 10)
):

    import matplotlib.pyplot as plt

    mutual_info = mutual_info.sort_values("total")
    mutual_info['total_rank'] = mutual_info.reset_index().index + 1
    mutual_info = mutual_info.sort_values("MI1")
    mutual_info['MI_rank'] = mutual_info.reset_index().index + 1
    mutual_info = mutual_info.sort_values("pct_pos_multiplier")
    mutual_info['pct_pos_multiplier_rank'] = mutual_info.reset_index().index + 1

    color_maps = {
        "grey": plt.cm.Greys,
        "purple": plt.cm.Purples,
        "blue": plt.cm.Blues,
        "green": plt.cm.Greens,
        "orange": plt.cm.Oranges,
        "red": plt.cm.Reds
    }

    f, ax = plt.subplots(figsize=figsize)
    mutual_info["size"] = mutual_info[size_col].map(
        lambda x: scale_range(x, mutual_info[size_col].min(), mutual_info[size_col].max(), 15, 25))
    mutual_info["color"] = mutual_info[color_col].map(
        lambda x: scale_range(x, mutual_info[color_col].min(), mutual_info[color_col].max(), .4, 1))
    mutual_info["color"] = mutual_info["color"].map(color_maps[color])

    mutual_info["x"] = mutual_info[x_col]
    mutual_info["y"] = mutual_info[y_col]
    ax.set_title(title)
    ax.set_xlim((mutual_info["x"].min() * .9, mutual_info["x"].max() * 1.1))
    ax.set_ylim((mutual_info["y"].min() * .9, mutual_info["y"].max() * 1.1))
    ax.set_ylabel("Mutual information")
    ax.set_xlabel("Likelihood ratio")

    texts = []
    for index, row in mutual_info.iterrows():
        texts.append(ax.text(row['x'], row['y'], index, size=row["size"], color=row["color"]))

    return ax
    # plt.savefig(value + '.pdf')