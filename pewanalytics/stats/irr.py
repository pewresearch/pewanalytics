import math
import scipy
import numpy as np
from nltk.metrics.agreement import AnnotationTask
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.inter_rater import fleiss_kappa
from pewtils import is_not_null


def kappa_sample_size_power(
    rate1, rate2, k1, k0, alpha=0.05, power=0.8, twosided=False
):

    """
    Translated from N.cohen.kappa: https://cran.r-project.org/web/packages/irr/irr.pdf

    :param rate1: the probability that the first rater will record a positive diagnosis
    :param rate2: the probability that the second rater will record a positive diagnosis
    :param k1: the true Cohen's Kappa statistic
    :param k0: the value of kappa under the null hypothesis
    :param alpha: type I error of test
    :param power: the desired power to detect the difference between true kappa and hypothetical kappa
    :param twosided: TRUE if test is two-sided
    :return: returns required sample size
    """

    if not twosided:
        d = 1
    else:
        d = 2
    pi_rater1 = 1 - rate1
    pi_rater2 = 1 - rate2
    pie = rate1 * rate2 + pi_rater1 * pi_rater2
    pi0 = k1 * (1 - pie) + pie
    pi22 = (pi0 - rate1 + pi_rater2) / 2
    pi11 = pi0 - pi22
    pi12 = rate1 - pi11
    pi21 = rate2 - pi11
    pi0h = k0 * (1 - pie) + pie
    pi22h = (pi0h - rate1 + pi_rater2) / 2
    pi11h = pi0h - pi22h
    pi12h = rate1 - pi11h
    pi21h = rate2 - pi11h
    Q = (1 - pie) ** (-4) * (
        pi11 * (1 - pie - (rate2 + rate1) * (1 - pi0)) ** 2
        + pi22 * (1 - pie - (pi_rater2 + pi_rater1) * (1 - pi0)) ** 2
        + (1 - pi0) ** 2
        * (pi12 * (rate2 + pi_rater1) ** 2 + pi21 * (pi_rater2 + rate1) ** 2)
        - (pi0 * pie - 2 * pie + pi0) ** 2
    )
    Qh = (1 - pie) ** (-4) * (
        pi11h * (1 - pie - (rate2 + rate1) * (1 - pi0h)) ** 2
        + pi22h * (1 - pie - (pi_rater2 + pi_rater1) * (1 - pi0h)) ** 2
        + (1 - pi0h) ** 2
        * (pi12h * (rate2 + pi_rater1) ** 2 + pi21h * (pi_rater2 + rate1) ** 2)
        - (pi0h * pie - 2 * pie + pi0h) ** 2
    )
    N = (
        (
            scipy.stats.norm.ppf(1 - alpha / d) * math.sqrt(Qh)
            + scipy.stats.norm.ppf(power) * math.sqrt(Q)
        )
        / (k1 - k0)
    ) ** 2
    return math.ceil(N)


def kappa_sample_size_CI(kappa0, kappaL, props, kappaU=None, alpha=0.05):

    """
    Translated from kappaSize: https://github.com/cran/kappaSize/blob/master/R/CIBinary.R

    :param kappa0: The preliminary value of kappa
    :param kappaL: The desired expected lower bound for a two-sided 100(1 - alpha) % confidence interval for kappa. Alternatively, if kappaU is set to NA, the procedure produces the number of required subjects for a one-sided confidence interval
    :param props: The anticipated prevalence of the desired trait
    :param kappaU: The desired expected upper confidence limit for kappa
    :param alpha: The desired type I error rate
    :return:
    """

    if not kappaU:
        chiCrit = scipy.stats.chi2.ppf((1 - 2 * alpha), 1)

    if kappaL and kappaU:
        chiCrit = scipy.stats.chi2.ppf((1 - alpha), 1)

    def CalcIT(rho0, rho1, Pi, n):
        def P0(r, p):
            x = (1 - p) ** 2 + r * p * (1 - p)
            return x

        def P1(r, p):
            x = 2 * (1 - r) * p * (1 - p)
            return x

        def P2(r, p):
            x = p ** 2 + r * p * (1 - p)
            return x

        r1 = ((n * P0(r=rho0, p=Pi)) - (n * P0(r=rho1, p=Pi))) ** 2 / (
            n * P0(r=rho1, p=Pi)
        )
        r2 = ((n * P1(r=rho0, p=Pi)) - (n * P1(r=rho1, p=Pi))) ** 2 / (
            n * P1(r=rho1, p=Pi)
        )
        r3 = ((n * P2(r=rho0, p=Pi)) - (n * P2(r=rho1, p=Pi))) ** 2 / (
            n * P2(r=rho1, p=Pi)
        )
        return sum([r for r in [r1, r2, r3] if r])

    n = 10
    result = 0
    while (result - 0.001) < chiCrit:
        result = CalcIT(kappa0, kappaL, props, n)
        n += 1
    return n


def compute_scores(
    coder_df,
    coder1,
    coder2,
    outcome_column,
    document_column,
    coder_column,
    weight_column,
    pos_label=None,
):
    """
    Computes a variety of IRR scores, including Cohen's kappa, Krippendorf's alpha, precision, and recall

    :param coder_df: A dataframe of codes
    :param coder1: The value in `coder_column` for rows corresponding to the first coder
    :param coder2: The value in `coder_column` for rows corresponding to the second coder
    :param outcome_column: The column that contains the codes
    :param document_column: The column that contains IDs for the documents
    :param coder_column: The column containing values that indicate which coder assigned the code
    :param weight_column: The column that contains sampling weights
    :param pos_label: The value indicating a positive label (optional)
    :return:
    """

    coder1_df = coder_df[coder_df[coder_column] == coder1]
    coder1_df.index = coder1_df[document_column]
    coder2_df = coder_df[coder_df[coder_column] == coder2]
    coder2_df.index = coder2_df[document_column]
    coder1_df = coder1_df[coder1_df.index.isin(coder2_df.index)]
    coder2_df = coder2_df[coder2_df.index.isin(coder1_df.index)].ix[coder1_df.index]

    row = {
        "coder1": coder1,
        "coder2": coder2,
        "n": len(coder1_df),
        "outcome_column": outcome_column,
        "pos_label": pos_label,
    }

    for labelsetname, labelset in [
        ("coder1", coder1_df[outcome_column]),
        ("coder2", coder2_df[outcome_column]),
    ]:

        if pos_label:
            labelset = (labelset == pos_label).astype(int)

        try:
            weighted_stats = DescrStatsW(labelset, weights=coder1_df[weight_column])
            unweighted_stats = DescrStatsW(labelset, weights=[1.0 for x in labelset])
            weighted_stats.mean
            unweighted_stats.mean
        except (TypeError, ValueError):
            try:
                weighted_stats = DescrStatsW(
                    labelset.astype(int), weights=coder1_df[weight_column]
                )
                unweighted_stats = DescrStatsW(
                    labelset.astype(int), weights=[1.0 for x in labelset]
                )
                weighted_stats.mean
                unweighted_stats.mean
            except (TypeError, ValueError):
                weighted_stats = None
                unweighted_stats = None

        if weighted_stats:
            row["{}_mean".format(labelsetname)] = weighted_stats.mean
            row["{}_std".format(labelsetname)] = weighted_stats.std_mean
        if unweighted_stats:
            row["{}_mean_unweighted".format(labelsetname)] = unweighted_stats.mean
            row["{}_std_unweighted".format(labelsetname)] = unweighted_stats.std_mean

    alpha = AnnotationTask(
        data=coder_df[[coder_column, document_column, outcome_column]].as_matrix()
    )
    try:
        alpha = alpha.alpha()
    except (ZeroDivisionError, ValueError):
        alpha = None
    row["alpha_unweighted"] = alpha

    labels = np.unique(coder_df[outcome_column])
    if len(labels) <= 2:

        row["cohens_kappa"] = cohen_kappa_score(
            coder1_df[outcome_column],
            coder2_df[outcome_column],
            sample_weight=coder1_df[weight_column],
            labels=labels,
        )

    try:
        row["accuracy"] = accuracy_score(
            coder1_df[outcome_column],
            coder2_df[outcome_column],
            sample_weight=coder1_df[weight_column],
        )
    except ValueError:
        row["accuracy"] = None

    try:
        row["f1"] = f1_score(
            coder1_df[outcome_column],
            coder2_df[outcome_column],
            pos_label=pos_label,
            sample_weight=coder1_df[weight_column],
            labels=labels,
        )
    except ValueError:
        row["f1"] = None

    try:
        row["precision"] = precision_score(
            coder1_df[outcome_column],
            coder2_df[outcome_column],
            pos_label=pos_label,
            sample_weight=coder1_df[weight_column],
            labels=labels,
        )
    except ValueError:
        row["precision"] = None

    try:
        row["recall"] = (
            recall_score(
                coder1_df[outcome_column],
                coder2_df[outcome_column],
                pos_label=pos_label,
                sample_weight=coder1_df[weight_column],
                labels=labels,
            ),
        )
    except ValueError:
        row["recall"] = None

    if is_not_null(row["precision"]) and is_not_null(row["recall"]):
        row["precision_recall_min"] = min([row["precision"], row["recall"]])
    else:
        row["precision_recall_min"] = None

    try:
        row["matthews_corrcoef"] = matthews_corrcoef(
            coder1_df[outcome_column],
            coder2_df[outcome_column],
            sample_weight=coder1_df[weight_column],
        )
    except ValueError:
        row["matthews_corrcoef"] = None

    try:
        row["roc_auc"] = (
            roc_auc_score(
                coder1_df[outcome_column],
                coder2_df[outcome_column],
                sample_weight=coder1_df[weight_column],
            )
            if len(np.unique(coder1_df[outcome_column])) > 1
            and len(np.unique(coder2_df[outcome_column])) > 1
            else None
        )
    except (ValueError, TypeError):
        row["roc_auc"] = None

    row["pct_agree_unweighted"] = np.average(
        [
            1 if c[0] == c[1] else 0
            for c in zip(coder1_df[outcome_column], coder2_df[outcome_column])
        ]
    )

    for k, v in row.items():
        if type(v) == tuple:
            row[k] = v[0]
            # For some weird reason, some of the sklearn scorers return 1-tuples sometimes

    return row


def compute_overall_scores(coder_df, document_column, outcome_column, coder_column):
    """
    Computes overall IRR scores
    
    :param coder_df: A dataframe of codes
    :param document_column: The column that contains IDs for the documents
    :param outcome_column: The column that contains the codes
    :param coder_column: The column containing values that indicate which coder assigned the code
    :return:
    """
    alpha = AnnotationTask(
        data=coder_df[[coder_column, document_column, outcome_column]].as_matrix()
    )
    try:
        alpha = alpha.alpha()
    except (ZeroDivisionError, ValueError):
        alpha = None

    grouped = coder_df.groupby(document_column).count()
    complete_docs = grouped[
        grouped[coder_column] == len(coder_df[coder_column].unique())
    ].index
    dataset = coder_df[coder_df[document_column].isin(complete_docs)]
    df = dataset.groupby([outcome_column, document_column]).count()[[coder_column]]
    df = df.unstack(outcome_column).fillna(0)

    if len(df) > 0:
        kappa = fleiss_kappa(df)
    else:
        kappa = None

    return {"alpha": alpha, "fleiss_kappa": kappa}
