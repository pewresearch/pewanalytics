from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
import random
import numpy as np


def compute_sample_weights_from_frame(frame, sample, weight_vars):

    """
    Takes two :py:class:`pandas.DataFrame` s and computes sampling weights for the second one, based on the first. \
    The first :py:class:`pandas.DataFrame` should be equivalent to the population that the second \
    :py:class:`pandas.DataFrame`, a sample, was drawn from. Weights will be calculated based on the differences in \
    the distribution of one or more variables specified in ``weight_vars`` (these should be the names of columns). \
    Returns a :py:class:`pandas.Series` equal in length to the ``sample`` with the computed weights.

    :param frame: :py:class:`pandas.DataFrame` (must contain all of the columns specified in ``weight_vars``)
    :param sample: :py:class:`pandas.DataFrame` (must contain all of the columns specified in ``weight_vars``)
    :param weight_vars: The names of the columns to use when computing weights.
    :type weight_vars: list
    :return: A :py:class:`pandas.Series` containing the weights for each row in the ``sample``

    Usage::

        from pewanalytics.stats.sampling import compute_sample_weights_from_frame
        import nltk
        import pandas as pd
        from sklearn.metrics.pairwise import linear_kernel
        from sklearn.feature_extraction.text import TfidfVectorizer

        nltk.download("inaugural")
        frame = pd.DataFrame([
            {"speech": fileid, "text": nltk.corpus.inaugural.raw(fileid)} for fileid in nltk.corpus.inaugural.fileids()
        ])
        # Let's grab a sample of speeches - some that mention specific terms, and an additional random sample
        frame['economy'] = frame['text'].str.contains("economy").astype(int)
        frame['health'] = frame['text'].str.contains("health").astype(int)
        frame['immigration'] = frame['text'].str.contains("immigration").astype(int)
        frame['education'] = frame['text'].str.contains("education").astype(int)
        sample = pd.concat([
            frame[frame['economy']==1].sample(5),
            frame[frame['health']==1].sample(5),
            frame[frame['immigration']==1].sample(5),
            frame[frame['education']==1].sample(5),
            frame.sample(5)
        ])
        # Now we can get the sampling weights to adjust it back to the population based on those variables

        >>> sample['weight'] = compute_sample_weights_from_frame(frame, sample, ["economy", "health", "immigration", "education"])
        >>> sample
                       speech                                               text  economy  health  immigration  education  count    weight
        7     1817-Monroe.txt  I should be destitute of feeling if I was not ...        1       1            0          0      1  1.005747
        11   1833-Jackson.txt  Fellow citizens, the will of the American peop...        1       0            0          0      1  2.370690
        34  1925-Coolidge.txt  My countrymen, no one can contemplate curre...           1       0            1          1      1  0.344828
        35    1929-Hoover.txt  My Countrymen: This occasion is not alone the ...        1       1            0          1      1  0.538793
        28  1901-McKinley.txt  My fellow-citizens, when we assembled here on ...        1       0            0          0      1  2.370690

    """

    if len(weight_vars) > 0:

        frame["count"] = 1
        sample["count"] = 1
        sample_grouped = sample.groupby(weight_vars).count()
        sample_grouped /= len(sample)
        frame_grouped = frame.groupby(weight_vars).count()
        frame_grouped /= len(frame)
        weights = frame_grouped / sample_grouped
        weights["weight"] = weights["count"]
        for c in weights.columns:
            if c not in weight_vars and c != "weight":
                del weights[c]
        try:
            sample = sample.merge(
                weights, how="left", left_on=weight_vars, right_index=True
            )
        except ValueError:
            weights = weights.reset_index()
            index = sample.index
            sample = sample.merge(
                weights, how="left", left_on=weight_vars, right_on=weight_vars
            )
            sample.index = index
    else:
        sample["weight"] = 1.0

    return sample["weight"]


def compute_balanced_sample_weights(sample, weight_vars, weight_column=None):

    """
    Takes a :py:class:`pandas.DataFrame` and one or more column names (``weight_vars``) and computes weights such \
    that every unique combination of values in the weighting columns are balanced (when weighted, the sum of the \
    observations with each combination will be equal to one another). Useful for balancing important groups in \
    training datasets, etc.

    :param sample: :py:class:`pandas.DataFrame` (must contain all of the columns specified in ``weight_vars``)
    :param weight_vars: The names of the columns to use when computing weights.
    :type weight_vars: list
    :param weight_column: An option column containing existing weights, which can be factored into the new weights.
    :type weight_column: str
    :return: A :py:class:`pandas.Series` containing the weights for each row in the ``sample``

    .. note:: All weight variables must be binary flags (1 or 0); if you want to weight using a non-binary variable, \
        you should convert it into a set of dummy variables and then pass those in as multiple columns.
        
    Usage::

        from pewanalytics.stats.sampling import compute_balanced_sample_weights
        import pandas as pd

        # Let's say we have a set of tweets from members of Congress
        df = pd.DataFrame([
            {"politician_id": 1, "party": "R", "tweet": "Example document"},
            {"politician_id": 1, "party": "R", "tweet": "Example document"},
            {"politician_id": 2, "party": "D", "tweet": "Example document"},
            {"politician_id": 2, "party": "D", "tweet": "Example document"},
            {"politician_id": 3, "party": "D", "tweet": "Example document"},
        ])
        df['is_republican'] = (df['party']=="R").astype(int)

        # We can balance the parties like so:

        >>> df['weight'] = compute_balanced_sample_weights(df, ["is_republican"])

        >>> df
           politician_id party             tweet  is_rep    weight  is_republican
        0              1     R  Example document       1  1.250000              1
        1              1     R  Example document       1  1.250000              1
        2              2     D  Example document       0  0.833333              0
        3              2     D  Example document       0  0.833333              0
        4              3     D  Example document       0  0.833333              0

    """

    if len(weight_vars) > 0:

        num_valid_combos = 0
        weight_vars = list(set(weight_vars))
        combo_weights = {}
        combos = list(
            set(
                [
                    tuple(row[weight_vars].values.astype(bool))
                    for index, row in sample.iterrows()
                ]
            )
        )
        for combo in combos:
            if weight_column:
                combo_weights[combo] = float(
                    sample[
                        eval(
                            " & ".join(
                                [
                                    "(sample['{}']=={})".format(col, c)
                                    for col, c in zip(weight_vars, combo)
                                ]
                            )
                        )
                    ][weight_column].sum()
                ) / float(sample[weight_column].sum())
            else:
                combo_weights[combo] = float(
                    len(
                        sample[
                            eval(
                                " & ".join(
                                    [
                                        "(sample['{}']=={})".format(col, c)
                                        for col, c in zip(weight_vars, combo)
                                    ]
                                )
                            )
                        ]
                    )
                ) / float(len(sample))
            if combo_weights[combo] > 0:
                num_valid_combos += 1
            else:
                del combo_weights[combo]

        balanced_ratio = 1.0 / float(num_valid_combos)
        combo_weights = {
            k: float(balanced_ratio) / float(v) for k, v in combo_weights.items()
        }

        sample["weight"] = sample.apply(
            lambda x: combo_weights[tuple([x[v] for v in weight_vars])], axis=1
        )

    else:
        sample["weight"] = 1.0

    return sample["weight"]


class SampleExtractor(object):
    """
    A helper class for extracting samples using various sampling methods.

    :param df: The sampling frame
    :type df: :py:class:`pandas.DataFrame`
    :param id_col: Column in the :py:class:`pandas.DataFrame` to be used as the unique ID of observations
    :type id_col: str
    :param verbose: Whether or not to print information during the sampling process (default=False)
    :type verbose: bool
    :param seed: Random seed (optional)
    :type seed: int

    """

    def __init__(self, df, id_col, verbose=False, seed=None):

        self.df = df
        self.id_col = id_col
        self.seed = seed
        self.verbose = verbose

        if not self.seed:
            self.seed = int(round(1000 * np.random.random()))

    def extract(self, sample_size, sampling_strategy="random", stratify_by=None):

        """
        Extract a sample from a :py:class:`pandas.DataFrame` using one of the following methods:

        - all: Returns all of the IDs
        - random: Returns a random sample
        - stratify: Proportional stratification, method from Kish, Leslie. "Survey sampling." (1965). Chapter 4.
        - stratify_even: Sample evenly from each strata (will obviously not be representative)
        - stratify_guaranteed: Proportional stratification, but the sample is guaranteed to contain at least one \
            observation from each strata (if sample size is small and/or there are many small strata, the resulting \
            sample may be far from representative)

        :param sample_size: The desired size of the sample
        :type sample_size: int
        :param sampling_strategy: The method to be used to extract samples. Options are: all, random, stratify, \
        stratify_even, stratify_guaranteed
        :type sampling_strategy: str
        :param stratify_by: Optional name of a column or list of columns in the :py:class:`pandas.DataFrame` to \
        stratify on
        :type stratify_by: str, list
        :return: A list of IDs reflecting the observations selected from the :py:class:`pandas.DataFrame` during \
        sampling
        :rtype: list

        Usage::

            from pewanalytics.stats.sampling import SampleExtractor
            import nltk
            import pandas as pd

            nltk.download("inaugural")
            frame = pd.DataFrame([
                {"speech": fileid, "text": nltk.corpus.inaugural.raw(fileid)} for fileid in nltk.corpus.inaugural.fileids()
            ])
            frame["century"] = frame['speech'].map(lambda x: "{}00".format(x.split("-")[0][:2]))

            >>> sampler = SampleExtractor(frame, "speech", seed=42)

            >>> sample_index = sampler.extract(12, sampling_strategy="random")
            frame[frame["speech"].isin(sample_index)]['century'].value_counts()
            1900    6
            1800    5
            1700    1
            Name: century, dtype: int64

            >>> sample_index = sampler.extract(12, sampling_strategy="stratify", stratify_by=['century'])
            frame[frame["speech"].isin(sample_index)]['century'].value_counts()
            1800    5
            1900    5
            2000    1
            1700    1
            Name: century, dtype: int64

            >>> sample_index = sampler.extract(12, sampling_strategy="stratify_even", stratify_by=['century'])
            frame[frame["speech"].isin(sample_index)]['century'].value_counts()
            1800    3
            2000    3
            1700    3
            1900    3
            Name: century, dtype: int64

            >>> sample_index = sampler.extract(12, sampling_strategy="stratify_guaranteed", stratify_by=['century'])
            frame[frame["speech"].isin(sample_index)]['century'].value_counts()
            1900    5
            1800    4
            1700    2
            2000    1
            Name: century, dtype: int64

        """

        strategies = [
            "all",
            "random",
            "stratify",
            "stratify_even",
            "stratify_guaranteed",
        ]
        if sampling_strategy not in strategies:
            raise Exception(
                "You must choose one of the following sampling strategies: {}".format(
                    strategies
                )
            )

        doc_ids = None

        if sampling_strategy == "all":
            doc_ids = self.df[self.id_col].values

        elif sampling_strategy == "random":
            doc_ids = self._random_sample(sample_size).values

        elif sampling_strategy.startswith("stratify"):

            if self.verbose:
                print("Stratify on columns: {}".format(",".join(stratify_by)))

            self.df["_stratify_by"] = (
                self.df[stratify_by].astype(str).apply("".join, axis=1)
            )

            # So you can pass in a decimal proportion of total dataframe or number of samples
            sample_n = (
                sample_size
                if sample_size >= 1
                else int(round(sample_size * self.df.shape[0]))
            )

            if sampling_strategy == "stratify":
                doc_ids = self._stratify_sample(sample_n)

            elif sampling_strategy == "stratify_even":
                doc_ids = self._stratify_even_sample(sample_n)

            elif sampling_strategy == "stratify_guaranteed":
                doc_ids = self._stratify_guaranteed_sample(sample_n)

            del self.df["_stratify_by"]

        if self.verbose:
            print("Sample of %i extracted" % (len(doc_ids)))

        return list(doc_ids)

    def _random_sample(self, sample_size):

        if self.verbose:
            print("Basic random sample")
        if sample_size >= 1:
            return self.df.sample(int(sample_size), random_state=self.seed)[self.id_col]
        else:
            return self.df.sample(frac=float(sample_size), random_state=self.seed)[
                self.id_col
            ]

    def _stratify_sample(self, sample_size):

        if self.verbose:
            print("Kish-style stratification")

        # Subset & copy cols that we care about
        data = self.df.copy()[[self.id_col] + ["_stratify_by"]]
        frame_size = data.shape[0]

        # Shuffle the dataframe
        if self.verbose:
            print("Random seed: {}".format(self.seed))
        np.random.seed(self.seed)
        if self.verbose:
            print("Dataframe before sorting: {}".format(data.head()))
        data.index = np.random.permutation(data.index)

        # Re-sort grouped by strata
        data = data.groupby("_stratify_by").apply(lambda x: x.sort_index())
        data.index = list(range(0, frame_size))
        if self.verbose:
            print("Dataframe after shuffle & groupby sorting: {}".format(data.head()))

        skip_interval = float(frame_size) / float(sample_size)

        start_index = np.random.uniform(0, skip_interval)  # index to start from
        if self.verbose:
            print("Start index: {}".format(start_index))
        sample_index = np.round(
            (np.zeros(sample_size) + start_index)
            + (np.arange(sample_size) * skip_interval)
        )

        # Return the real id column
        sample_ids = data[data.index.isin(sample_index)][self.id_col].values

        return sample_ids

    def _stratify_even_sample(self, sample_size):

        random.seed(self.seed)
        docs_per_strata = int(
            float(sample_size)
            / float(self.df.groupby("_stratify_by")[self.id_col].count().count())
        )
        if self.verbose:
            print(
                "Drawing even samples of {} across all stratification groups".format(
                    docs_per_strata
                )
            )
        doc_ids = []
        for strata in self.df["_stratify_by"].unique():
            strata_data = self.df[self.df["_stratify_by"] == strata]
            doc_ids.extend(
                list(strata_data.sample(docs_per_strata)[self.id_col].values)
            )
        if len(doc_ids) < sample_size:
            doc_ids.extend(
                list(self.df.sample(sample_size - len(doc_ids))[self.id_col].values)
            )

        return doc_ids

    def _stratify_guaranteed_sample(self, sample_size):

        # Number of groups to stratify by must be less than the total sample size
        strata_groups = self.df.groupby("_stratify_by")[self.id_col].count().count()
        if sample_size > strata_groups:
            if self.verbose:
                print(
                    "Sampling one document per strata first ({} strata total)".format(
                        strata_groups
                    )
                )
            strata_one = (
                self.df.groupby("_stratify_by")
                .apply(lambda x: x.sample(1, random_state=self.seed))[self.id_col]
                .values
            )
        else:
            if self.verbose:
                print(
                    "There are more strata groups ({}) than things to sample: {}".format(
                        strata_groups, sample_size
                    )
                )
            strata_one = (
                self.df.groupby("_stratify_by")
                .apply(lambda x: x.sample(1, random_state=self.seed))[self.id_col]
                .sample(sample_size)
                .values
            )

        left_to_sample = sample_size - len(strata_one)
        if left_to_sample > 0:
            doc_ids = SampleExtractor(
                self.df[~self.df[self.id_col].isin(strata_one)],
                self.id_col,
                seed=self.seed,
                verbose=self.verbose,
            )._stratify_sample(left_to_sample)
            doc_ids = list(doc_ids) + list(strata_one)
        else:
            if self.verbose:
                print("Nothing left to sample, no stratification applied")
            doc_ids = strata_one

        return doc_ids
