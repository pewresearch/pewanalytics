"""

"""
from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
import random
import numpy as np
import pandas as pd


def compute_sample_weights_from_frame(frame, sample, weight_vars):

    """
    Takes two dataframes and computes sampling weights for the second one, based on the first. The first dataframe
    should be equivalent to the population that the second dataframe, a sample, was drawn from. Weights will be
    calculated based on the differences in the distribution of one or more variables specified in `weight_vars`
    (these should be the names of columns). Returns a series equal in length to the `sample` with the computed weights.

    :param frame: DataFrame (must contain all of the columns specified in `weight_vars`)
    :param sample: DataFrame (must contain all of the columns specified in `weight_vars`)
    :param weight_vars: The names of the columns to use when computing weights.
    :return: A Series containing the weights for each row in the `sample`
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
    Takes a DataFrame and one or more column names (`weight_vars`) and computes weights such that every unique
    combination of values in the weighting columns are balanced (when weighted, the sum of the observations with each
    combination will be equal to one another.)

    :param sample: DataFrame (must contain all of the columns specified in `weight_vars`)
    :param weight_vars: The names of the columns to use when computing weights.
    :param weight_column: An option column containing existing weights, which can be factored into the new weights.
    :return:
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
    A helper class for extracting samples using various sampling methods. After specifying the sampling
    options on the SampleExtractor, you can make multiple calls to `extract`, passing in a DataFrame from
    which to sample, and the desired size of the sample. The available sampling methods are:

    - all: Returns all of the IDs
    - random: Returns a random sample
    - stratify: Proportional stratification, method from Kish, Leslie. "Survey sampling." (1965). Chapter 4.
    - stratify_even: Sample evenly from each strata (will obviously not be representative)
    - stratify_guaranteed: Proportional stratification, but the sample is guaranteed to contain at least one
    observation from each strata (if sample size is small and/or there are many small strata, the resulting sample
    may be far from representative)

    :param sampling_strategy: The method to be used to extract samples. Options are: all, random, stratify,
    stratify_even, stratify_guaranteed
    :param stratify_by: Optional name of a column or list of columns in the DataFrame to stratify on
    :param id_col: column in the DataFrame to be used as the unique ID of observations
    :param verbose: Whether or not to print information during the sampling process (default=False)
    :param seed: Random seed (optional)
    """
    def __init__(
        self,
        sampling_strategy="random",
        stratify_by=None,
        id_col=None,
        verbose=False,
        seed=None,
    ):
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

        self.stratify_by = stratify_by
        self.id_col = id_col
        self.seed = seed
        self.sampling_strategy = sampling_strategy
        self.verbose = verbose

        if not self.seed:
            self.seed = int(round(1000 * np.random.random()))

    def extract(self, df, sample_size):

        """
        Extract a sample from a DataFrame

        :param df: The DataFrame to sample from
        :param sample_size: The desired size of the sample
        :return: A list of IDs reflecting the observations selected from the DataFrame during sampling
        """

        doc_ids = None

        if self.sampling_strategy == "all":
            doc_ids = df[self.id_col].values

        elif self.sampling_strategy == "random":
            doc_ids = self._random_sample(df[self.id_col], sample_size).values

        elif self.sampling_strategy.startswith("stratify"):

            if self.verbose:
                print("Stratify on columns: {}".format(",".join(self.stratify_by)))

            df["_stratify_by"] = df[self.stratify_by].astype(str).apply("".join, axis=1)

            # So you can pass in a decimal proportion of total dataframe or number of samples
            sample_n = (
                sample_size
                if sample_size >= 1
                else int(round(sample_size * df.shape[0]))
            )

            if self.sampling_strategy == "stratify":
                doc_ids = self._stratify_sample(df, sample_n)

            elif self.sampling_strategy == "stratify_even":
                doc_ids = self._stratify_even_sample(df, sample_n)

            elif self.sampling_strategy == "stratify_guaranteed":
                doc_ids = self._stratify_guaranteed_sample(df, sample_n)

            del df["_stratify_by"]

        if self.verbose:
            print("Sample of %i extracted" % (len(doc_ids)))

        return list(doc_ids)

    def _random_sample(self, df, sample_size):

        if self.verbose:
            print("Basic random sample")
        if sample_size >= 1:
            try:
                return df.sample(int(sample_size), random_state=self.seed)
            except ValueError:
                return df
        else:
            return df.sample(frac=float(sample_size), random_state=self.seed)

    def _stratify_sample(self, df, sample_n):

        if self.verbose:
            print("Kish-style stratification")

        # Subset & copy cols that we care about
        data = df.copy()[[self.id_col] + ["_stratify_by"]]
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

        skip_interval = float(frame_size) / float(sample_n)

        start_index = np.random.uniform(0, skip_interval)  # index to start from
        if self.verbose:
            print("Start index: {}".format(start_index))
        sample_index = np.round(
            (np.zeros(sample_n) + start_index) + (np.arange(sample_n) * skip_interval)
        )

        # Return the real id column
        sample_ids = data[data.index.isin(sample_index)][self.id_col].values

        return sample_ids

    def _stratify_even_sample(self, df, sample_size):

        random.seed(self.seed)
        docs_per_strata = int(
            float(sample_size)
            / float(df.groupby("_stratify_by")[self.id_col].count().count())
        )
        if self.verbose:
            print(
                "Drawing even samples of {} across all stratification groups".format(
                    docs_per_strata
                )
            )
        doc_ids = []
        for strata in df["_stratify_by"].unique():
            strata_data = df[df["_stratify_by"] == strata]
            doc_ids.extend(
                list(strata_data.sample(docs_per_strata)[self.id_col].values)
            )
        if len(doc_ids) < sample_size:
            doc_ids.extend(
                list(df.sample(sample_size - len(doc_ids))[self.id_col].values)
            )

        return doc_ids

    def _stratify_guaranteed_sample(self, df, sample_size):

        # Number of groups to stratify by must be less than the total sample size
        strata_groups = df.groupby("_stratify_by")[self.id_col].count().count()
        if sample_size > strata_groups:
            if self.verbose:
                print(
                    "Sampling one document per strata first ({} strata total)".format(
                        strata_groups
                    )
                )
            strata_one = (
                df.groupby("_stratify_by")
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
                df.groupby("_stratify_by")
                .apply(lambda x: x.sample(1, random_state=self.seed))[self.id_col]
                .sample(sample_size)
                .values
            )

        left_to_sample = sample_size - len(strata_one)
        if left_to_sample > 0:
            doc_ids = self._stratify_sample(
                df[~df[self.id_col].isin(strata_one)], left_to_sample
            )
            doc_ids = list(doc_ids) + list(strata_one)
        else:
            if self.verbose:
                print("Nothing left to sample, no stratification applied")
            doc_ids = strata_one

        return doc_ids
