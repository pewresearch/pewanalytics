import math
import copy
import scipy
import numpy as np
import pandas as pd
from nltk.metrics import masi_distance, jaccard_distance
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
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.inter_rater import fleiss_kappa
from pewtils import is_not_null
from scipy import spatial


def kappa_sample_size_power(
    rate1, rate2, k1, k0, alpha=0.05, power=0.8, twosided=False
):

    """
    Python translation of the ``N.cohen.kappa`` function from the ``irr`` R package.

    Source: https://cran.r-project.org/web/packages/irr/irr.pdf

    :param rate1: The probability that the first rater will record a positive diagnosis
    :type rate1: float
    :param rate2: The probability that the second rater will record a positive diagnosis
    :type rate2: float
    :param k1: The true Cohen's Kappa statistic
    :type k1: float
    :param k0: The value of kappa under the null hypothesis
    :type k0: float
    :param alpha: Type I error of test
    :type alpha: float
    :param power: The desired power to detect the difference between true kappa and hypothetical kappa
    :type power: float
    :param twosided: Set this to ``True`` if the test is two-sided
    :param twosided: bool
    :return: Returns the required sample size
    :rtype: int

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

    Helps determine the required document sample size to confirm that Cohen's Kappa between coders is at or \
    above a minimum threhsold. Useful in situations where multiple coders code a set of documents for a binary outcome.

    This function takes the observed kappa and proportion of positive cases from the sample, along with a lower-bound \
    for the minimum acceptable kappa, and returns the sample size required to confirm that the coders' agreement is \
    truly above that minimum level of kappa with 95% certainty. If the current sample size is below the required \
    sample size returned by this function, it can provide a rough estimate of how many additional documents need \
    to be coded - assuming that the coders continue agreeing and observing positive cases at the same rate.

    Translated from the ``kappaSize`` R package, ``CIBinary``: \
    https://github.com/cran/kappaSize/blob/master/R/CIBinary.R

    :param kappa0: The preliminary value of kappa
    :param kappa0: float
    :param kappaL: The desired expected lower bound for a two-sided 100(1 - alpha) % confidence interval for kappa. \
    Alternatively, if kappaU is set to NA, the procedure produces the number of required subjects for a one-sided \
    confidence interval
    :type kappaL: float
    :param props: The anticipated prevalence of the desired trait
    :type props: float
    :param kappaU: The desired expected upper confidence limit for kappa
    :type kappaU: float
    :param alpha: The desired type I error rate
    :type alpha: float
    :return: Returns the required sample size

    Usage::

        from pewanalytics.stats.irr import kappa_sample_size_CI

        observed_kappa = 0.8
        desired_kappa = 0.7
        observed_proportion = 0.5

        >>> kappa_sample_size(observed_kappa, desired_kappa, observed_proportion)
        140

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
    weight_column=None,
    pos_label=None,
):

    """
    Computes a variety of inter-rater reliability scores, including Cohen's kappa, Krippendorf's alpha, precision,
    and recall. The input data must consist of a :py:class:`pandas.DataFrame` with the following columns:

        - A column with values that indicate the coder (like a name)
        - A column with values that indicate the document (like an ID)
        - A column with values that indicate the code value
        - (Optional) A column with document weights

    This function will return a :py:class:`pandas.DataFrame` with agreement scores between the two specified coders.

    :param coder_df: A :py:class:`pandas.DataFrame` of codes
    :type coder_df: :py:class:`pandas.DataFrame`
    :param coder1: The value in ``coder_column`` for rows corresponding to the first coder
    :type coder1: str or int
    :param coder2: The value in ``coder_column`` for rows corresponding to the second coder
    :type coder2: str or int
    :param outcome_column: The column that contains the codes
    :type outcome_column: str
    :param document_column: The column that contains IDs for the documents
    :type document_column: str
    :param coder_column: The column containing values that indicate which coder assigned the code
    :type coder_column: str
    :param weight_column: The column that contains sampling weights
    :type weight_column: str
    :param pos_label: The value indicating a positive label (optional)
    :type pos_label: str or int
    :return: A dictionary of scores
    :rtype: dict

    .. note:: If using a multi-class (non-binary) code, some scores may come back null or not compute as expected. \
        We recommend running the function separately for each specific code value as a binary flag by providing \
        each unique value to the ``pos_label`` argument. If ``pos_label`` is not provided for multi-class codes, \
        this function will attempt to compute scores based on support-weighted averages.

    Usage::

        from pewanalytics.stats.irr import compute_scores
        import pandas as pd

        df = pd.DataFrame([
            {"coder": "coder1", "document": 1, "code": "2"},
            {"coder": "coder2", "document": 1, "code": "2"},
            {"coder": "coder1", "document": 2, "code": "1"},
            {"coder": "coder2", "document": 2, "code": "2"},
            {"coder": "coder1", "document": 3, "code": "0"},
            {"coder": "coder2", "document": 3, "code": "0"},
        ])

        >>> compute_scores(df, "coder1", "coder2", "code", "document", "coder")
        {'coder1': 'coder1',
         'coder2': 'coder2',
         'n': 3,
         'outcome_column': 'code',
         'pos_label': None,
         'coder1_mean_unweighted': 1.0,
         'coder1_std_unweighted': 0.5773502691896257,
         'coder2_mean_unweighted': 1.3333333333333333,
         'coder2_std_unweighted': 0.6666666666666666,
         'alpha_unweighted': 0.5454545454545454,
         'accuracy': 0.6666666666666666,
         'f1': 0.5555555555555555,
         'precision': 0.5,
         'recall': 0.6666666666666666,
         'precision_recall_min': 0.5,
         'matthews_corrcoef': 0.6123724356957946,
         'roc_auc': None,
         'pct_agree_unweighted': 0.6666666666666666}

        >>> compute_scores(df, "coder1", "coder2", "code", "document", "coder", pos_label="0")
         {'coder1': 'coder1',
         'coder2': 'coder2',
         'n': 3,
         'outcome_column': 'code',
         'pos_label': '0',
         'coder1_mean_unweighted': 0.3333333333333333,
         'coder1_std_unweighted': 0.3333333333333333,
         'coder2_mean_unweighted': 0.3333333333333333,
         'coder2_std_unweighted': 0.3333333333333333,
         'alpha_unweighted': 1.0,
         'cohens_kappa': 1.0,
         'accuracy': 1.0,
         'f1': 1.0,
         'precision': 1.0,
         'recall': 1.0,
         'precision_recall_min': 1.0,
         'matthews_corrcoef': 1.0,
         'roc_auc': 1.0,
         'pct_agree_unweighted': 1.0}

        >>> compute_scores(df, "coder1", "coder2", "code", "document", "coder", pos_label="1")
        {'coder1': 'coder1',
         'coder2': 'coder2',
         'n': 3,
         'outcome_column': 'code',
         'pos_label': '1',
         'coder1_mean_unweighted': 0.3333333333333333,
         'coder1_std_unweighted': 0.3333333333333333,
         'coder2_mean_unweighted': 0.0,
         'coder2_std_unweighted': 0.0,
         'alpha_unweighted': 0.0,
         'cohens_kappa': 0.0,
         'accuracy': 0.6666666666666666,
         'f1': 0.0,
         'precision': 0.0,
         'recall': 0.0,
         'precision_recall_min': 0.0,
         'matthews_corrcoef': 1.0,
         'roc_auc': None,
         'pct_agree_unweighted': 0.6666666666666666}

        >>> compute_scores(df, "coder1", "coder2", "code", "document", "coder", pos_label="2")
        {'coder1': 'coder1',
         'coder2': 'coder2',
         'n': 3,
         'outcome_column': 'code',
         'pos_label': '2',
         'coder1_mean_unweighted': 0.3333333333333333,
         'coder1_std_unweighted': 0.3333333333333333,
         'coder2_mean_unweighted': 0.6666666666666666,
         'coder2_std_unweighted': 0.3333333333333333,
         'alpha_unweighted': 0.4444444444444444,
         'cohens_kappa': 0.3999999999999999,
         'accuracy': 0.6666666666666666,
         'f1': 0.6666666666666666,
         'precision': 0.5,
         'recall': 1.0,
         'precision_recall_min': 0.5,
         'matthews_corrcoef': 0.5,
         'roc_auc': 0.75,
         'pct_agree_unweighted': 0.6666666666666666}


    """

    old_np_settings = np.seterr(all="raise")

    coder_df = copy.deepcopy(coder_df)
    if pos_label:
        coder_df[outcome_column] = (coder_df[outcome_column] == pos_label).astype(int)
    coder1_df = coder_df[coder_df[coder_column] == coder1]
    coder1_df.index = coder1_df[document_column]
    coder2_df = coder_df[coder_df[coder_column] == coder2]
    coder2_df.index = coder2_df[document_column]
    coder1_df = coder1_df[coder1_df.index.isin(coder2_df.index)]
    coder2_df = coder2_df[coder2_df.index.isin(coder1_df.index)].loc[coder1_df.index]

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

        if weight_column:
            try:
                weighted_stats = DescrStatsW(labelset, weights=coder1_df[weight_column])
                if weighted_stats:
                    row["{}_mean".format(labelsetname)] = weighted_stats.mean
                    row["{}_std".format(labelsetname)] = weighted_stats.std_mean
            except (TypeError, ValueError):
                try:
                    weighted_stats = DescrStatsW(
                        labelset.astype(int), weights=coder1_df[weight_column]
                    )
                    if weighted_stats:
                        row["{}_mean".format(labelsetname)] = weighted_stats.mean
                        row["{}_std".format(labelsetname)] = weighted_stats.std_mean
                except (TypeError, ValueError):
                    pass

        try:
            unweighted_stats = DescrStatsW(labelset, weights=[1.0 for x in labelset])
            if unweighted_stats:
                row["{}_mean_unweighted".format(labelsetname)] = unweighted_stats.mean
                row[
                    "{}_std_unweighted".format(labelsetname)
                ] = unweighted_stats.std_mean
        except (TypeError, ValueError):
            try:
                unweighted_stats = DescrStatsW(
                    labelset.astype(int), weights=[1.0 for x in labelset]
                )
                if unweighted_stats:
                    row[
                        "{}_mean_unweighted".format(labelsetname)
                    ] = unweighted_stats.mean
                    row[
                        "{}_std_unweighted".format(labelsetname)
                    ] = unweighted_stats.std_mean
            except (TypeError, ValueError):
                pass

    alpha = AnnotationTask(
        data=coder_df[[coder_column, document_column, outcome_column]].values
    )
    try:
        alpha = alpha.alpha()
    except (ZeroDivisionError, ValueError):
        alpha = None
    row["alpha_unweighted"] = alpha

    labels = np.unique(coder_df[outcome_column])
    if len(labels) <= 2:

        try:
            row["cohens_kappa"] = cohen_kappa_score(
                coder1_df[outcome_column],
                coder2_df[outcome_column],
                sample_weight=coder1_df[weight_column] if weight_column else None,
                labels=labels,
            )
        except FloatingPointError:
            row["cohens_kappa"] = 1.0

    try:
        row["accuracy"] = accuracy_score(
            coder1_df[outcome_column],
            coder2_df[outcome_column],
            sample_weight=coder1_df[weight_column] if weight_column else None,
        )
    except ValueError:
        row["accuracy"] = None

    try:
        row["f1"] = f1_score(
            coder1_df[outcome_column],
            coder2_df[outcome_column],
            sample_weight=coder1_df[weight_column] if weight_column else None,
            labels=labels,
            average="weighted" if not pos_label else "binary",
        )
    except ValueError:
        row["f1"] = None

    try:
        row["precision"] = precision_score(
            coder1_df[outcome_column],
            coder2_df[outcome_column],
            sample_weight=coder1_df[weight_column] if weight_column else None,
            labels=labels,
            average="weighted" if not pos_label else "binary",
        )
    except ValueError:
        row["precision"] = None

    try:
        row["recall"] = recall_score(
            coder1_df[outcome_column],
            coder2_df[outcome_column],
            sample_weight=coder1_df[weight_column] if weight_column else None,
            labels=labels,
            average="weighted" if not pos_label else "binary",
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
            sample_weight=coder1_df[weight_column] if weight_column else None,
        )
    except ValueError:
        row["matthews_corrcoef"] = None
    except FloatingPointError:
        row["matthews_corrcoef"] = 1.0
    if row["accuracy"] == 1.0:
        # In newer versions of sklearn, perfect agreement returns zero incorrectly
        # We'll use the accuracy score to detect perfect agreement and correct it:
        row["matthews_corrcoef"] = 1.0

    try:

        row["roc_auc"] = (
            roc_auc_score(
                coder1_df[outcome_column],
                coder2_df[outcome_column],
                sample_weight=coder1_df[weight_column] if weight_column else None,
                labels=labels,
                average="weighted" if not pos_label else None,
            )
            if len(np.unique(coder1_df[outcome_column])) > 1
            and len(np.unique(coder2_df[outcome_column])) > 1
            else None
        )
    except TypeError:
        try:
            row["roc_auc"] = (
                roc_auc_score(
                    coder1_df[outcome_column],
                    coder2_df[outcome_column],
                    sample_weight=coder1_df[weight_column] if weight_column else None,
                    average="weighted" if not pos_label else None,
                )
                if len(np.unique(coder1_df[outcome_column])) > 1
                and len(np.unique(coder2_df[outcome_column])) > 1
                else None
            )
        except (ValueError, TypeError):
            row["roc_auc"] = None
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

    np.seterr(**old_np_settings)

    return row


def compute_overall_scores(coder_df, outcome_column, document_column, coder_column):

    """
    Computes overall inter-rater reliability scores (Krippendorf's Alpha and Fleiss' Kappa). Allows for more than two \
    coders and code values. The input data must consist of a :py:class:`pandas.DataFrame` with the following columns:

        - A column with values that indicate the coder (like a name)
        - A column with values that indicate the document (like an ID)
        - A column with values that indicate the code value

    :param coder_df: A :py:class:`pandas.DataFrame` of codes
    :type coder_df: :py:class:`pandas.DataFrame`
    :param outcome_column: The column that contains the codes
    :type outcome_column: str
    :param document_column: The column that contains IDs for the documents
    :type document_column: str
    :param coder_column: The column containing values that indicate which coder assigned the code
    :type coder_column: str
    :return: A dictionary containing the scores
    :rtype: dict

    Usage::

        from pewanalytics.stats.irr import compute_overall_scores
        import pandas as pd

        df = pd.DataFrame([
            {"coder": "coder1", "document": 1, "code": "2"},
            {"coder": "coder2", "document": 1, "code": "2"},
            {"coder": "coder1", "document": 2, "code": "1"},
            {"coder": "coder2", "document": 2, "code": "2"},
            {"coder": "coder1", "document": 3, "code": "0"},
            {"coder": "coder2", "document": 3, "code": "0"},
        ])

        >>> compute_overall_scores(df, "code", "document", "coder")
        {'alpha': 0.5454545454545454, 'fleiss_kappa': 0.4545454545454544}

    """

    alpha = AnnotationTask(
        data=coder_df[[coder_column, document_column, outcome_column]].values
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


def compute_overall_scores_multivariate(
    coder_df, document_column, coder_column, outcome_columns
):
    """
    Computes overall inter-rater reliability scores (Krippendorf's Alpha and Fleiss' Kappa). Allows for more than two \
    coders, code values, AND variables. All variables and values will be converted into a matrix of dummy variables, \
    and Alpha and Kappa will be computed using four different distance metrics:

        - Discrete agreement (exact agreement across all outcome columns)
        - Jaccard coefficient
        - MASI distance
        - Cosine similarity

    The input data must consist of a :py:class:`pandas.DataFrame` with the following \
    columns:

        - A column with values that indicate the coder (like a name)
        - A column with values that indicate the document (like an ID)
        - One or more columns with values that indicate code values

    This code was adapted from a very helpful StackExchange post:
    https://stats.stackexchange.com/questions/511927/interrater-reliability-with-multi-rater-multi-label-dataset

    :param coder_df: A :py:class:`pandas.DataFrame` of codes
    :type coder_df: :py:class:`pandas.DataFrame`
    :param document_column: The column that contains IDs for the documents
    :type document_column: str
    :param coder_column: The column containing values that indicate which coder assigned the code
    :type coder_column: str
    :param outcome_columns: The columns that contains the codes
    :type outcome_columns: list
    :return: A dictionary containing the scores
    :rtype: dict

    Usage::

        from pewanalytics.stats.irr import compute_overall_scores_multivariate
        import pandas as pd

        coder_df = pd.DataFrame([
            {"coder": "coder1", "document": 1, "code": "2"},
            {"coder": "coder2", "document": 1, "code": "2"},
            {"coder": "coder1", "document": 2, "code": "1"},
            {"coder": "coder2", "document": 2, "code": "2"},
            {"coder": "coder1", "document": 3, "code": "0"},
            {"coder": "coder2", "document": 3, "code": "0"},
        ])

        >>> compute_overall_scores_multivariate(coder_df, 'document', 'coder', ["code"])
        {'fleiss_kappa_discrete': 0.4545454545454544,
         'fleiss_kappa_jaccard': 0.49999999999999994,
         'fleiss_kappa_masi': 0.49999999999999994,
         'fleiss_kappa_cosine': 0.49999999999999994,
         'alpha_discrete': 0.5454545454545454,
         'alpha_jaccard': 0.5454545454545454,
         'alpha_masi': 0.5454545454545454,
         'alpha_cosine': 0.5454545454545454}

        coder_df = pd.DataFrame([
            {"coder": "coder1", "document": 1, "code1": "2", "code2": "1"},
            {"coder": "coder2", "document": 1, "code1": "2", "code2": "1"},
            {"coder": "coder1", "document": 2, "code1": "1", "code2": "0"},
            {"coder": "coder2", "document": 2, "code1": "2", "code2": "1"},
            {"coder": "coder1", "document": 3, "code1": "0", "code2": "0"},
            {"coder": "coder2", "document": 3, "code1": "0", "code2": "0"},
        ])

        >>> compute_overall_scores_multivariate(coder_df, 'document', 'coder', ["code1", "code2"])
        {'fleiss_kappa_discrete': 0.4545454545454544,
         'fleiss_kappa_jaccard': 0.49999999999999994,
         'fleiss_kappa_masi': 0.49999999999999994,
         'fleiss_kappa_cosine': 0.49999999999999994,
         'alpha_discrete': 0.5454545454545454,
         'alpha_jaccard': 0.5161290322580645,
         'alpha_masi': 0.5361781076066792,
         'alpha_cosine': 0.5}

    """

    dummies = pd.concat(
        [pd.get_dummies(coder_df[c], prefix=c) for c in outcome_columns], axis=1
    )
    outcome_columns = dummies.columns
    coder_df = pd.concat([coder_df, dummies], axis=1)
    coder_df = coder_df.dropna(subset=outcome_columns)
    coder_df["value"] = coder_df.apply(
        lambda x: frozenset([col for col in outcome_columns if int(x[col]) == 1]),
        axis=1,
    )

    discrete_task = AnnotationTask(
        data=coder_df[[coder_column, document_column, "value"]].values
    )
    kappa_discrete = discrete_task.multi_kappa()
    alpha_discrete = discrete_task.alpha()

    task_data = coder_df[[coder_column, document_column, "value"]].values
    task_data = [tuple(t) for t in task_data]

    jaccard_task = AnnotationTask(data=task_data, distance=jaccard_distance)
    masi_task = AnnotationTask(data=task_data, distance=masi_distance)

    mlb = MultiLabelBinarizer()
    task_data = [
        (r[0][0], r[0][1], tuple(r[1]))
        for r in zip(task_data, mlb.fit_transform([t[2] for t in task_data]))
    ]
    cosine_task = AnnotationTask(data=task_data, distance=spatial.distance.cosine)

    kappa_jaccard = jaccard_task.multi_kappa()
    kappa_masi = masi_task.multi_kappa()
    kappa_cosine = cosine_task.multi_kappa()
    alpha_jaccard = jaccard_task.alpha()
    alpha_masi = masi_task.alpha()
    alpha_cosine = cosine_task.alpha()

    return {
        "fleiss_kappa_discrete": kappa_discrete,
        "fleiss_kappa_jaccard": kappa_jaccard,
        "fleiss_kappa_masi": kappa_masi,
        "fleiss_kappa_cosine": kappa_cosine,
        "alpha_discrete": alpha_discrete,
        "alpha_jaccard": alpha_jaccard,
        "alpha_masi": alpha_masi,
        "alpha_cosine": alpha_cosine,
    }
