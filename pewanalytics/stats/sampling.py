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
    :frame: DataFrame
    :sample: Number of dimensions you want
    :weight_vars:
    :return:
    """

    if len(weight_vars) > 0:

        frame['count'] = 1
        sample['count'] = 1
        sample_grouped = sample.groupby(weight_vars).count()
        sample_grouped /= len(sample)
        frame_grouped = frame.groupby(weight_vars).count()
        frame_grouped /= len(frame)
        weights = frame_grouped / sample_grouped
        weights["weight"] = weights['count']
        for c in weights.columns:
            if c not in weight_vars and c != "weight":
                del weights[c]
        try: sample = sample.merge(weights, how="left", left_on=weight_vars, right_index=True)
        except ValueError:
            weights = weights.reset_index()
            index = sample.index
            sample = sample.merge(weights, how="left", left_on=weight_vars, right_on=weight_vars)
            sample.index = index
    else:
        sample['weight'] = 1.0

    return sample['weight']


def compute_balanced_sample_weights(sample, weight_vars, weight_column=None):

    """
    :sample: Number of dimensions you want
    :weight_vars:
    :return:
    """

    if len(weight_vars) > 0:

        num_valid_combos = 0
        weight_vars = list(set(weight_vars))
        combo_weights = {}
        combos = list(set([tuple(row[weight_vars].values.astype(bool)) for index, row in sample.iterrows()]))
        # for combo in itertools.product([True, False], repeat=len(weight_vars)):
        for combo in combos:
            if weight_column:
                combo_weights[combo] = float(sample[eval(" & ".join(["(sample['{}']=={})".format(col, c) for col, c in zip(weight_vars, combo)]))][weight_column].sum()) / float(sample[weight_column].sum())
            else:
                combo_weights[combo] = float(len(sample[eval(" & ".join(["(sample['{}']=={})".format(col, c) for col, c in zip(weight_vars, combo)]))])) / float(len(sample))
            if combo_weights[combo] > 0:
                num_valid_combos += 1
            else:
                del combo_weights[combo]

        balanced_ratio = 1.0 / float(num_valid_combos)
        combo_weights = {k: float(balanced_ratio) / float(v) for k, v in combo_weights.iteritems()}

        sample['weight'] = sample.apply(lambda x: combo_weights[tuple([x[v] for v in weight_vars])], axis=1)

    else:
        sample['weight'] = 1.0

    return sample['weight']


class SampleExtractor(object):

    def __init__(self, sampling_strategy='random', stratify_by=None, id_col=None, logger=None, seed=None):

        """
        :param df: dataframe
        :param sample_size:  integer (for sample size) or decimal (percentage of df)
        :param stratify_by: (column or list of columns in the dataframe to stratify on)
        :param id_col: column in the dataframe to be used as the record id
        :param seed: random seed ( optional )
        :param type: type of stratification to use . pass "even" if sample evenly from each strata. Otherwise strata-proportional sampling will be applied.
        :return:
        """

        strategies = ["all", "random", "stratify", "stratify_even", "stratify_guaranteed", "stratify_alt"]
        if sampling_strategy not in strategies:
            raise Exception("You must choose one of the following sampling strategies: {}".format(strategies))

        self.stratify_by = stratify_by
        self.id_col = id_col
        self.seed = seed
        self.sampling_strategy = sampling_strategy
        self.logger = logger

        if not self.seed:
            self.seed = int(round(1000 * np.random.random()))

    def extract(self, df, sample_size):

        if self.sampling_strategy == "all":

            doc_ids = df[self.id_col].values

        elif self.sampling_strategy == "random":

            # If no stratification at all
            if self.logger:
                self.logger.info("Basic random sample")
            doc_ids = self._basic_sample(df[self.id_col], sample_size).values

        elif self.sampling_strategy.startswith("stratify"):

            if self.logger:
                print("Stratify on columns: {}".format(",".join(self.stratify_by)))
            df['_stratify_by'] = df[self.stratify_by].astype(str).apply(''.join, axis=1)
            frame_size = df.shape[0]
            # So you can pass in a decimal proportion of total dataframe or number of samples
            sample_n = sample_size if sample_size >= 1 else int(round(sample_size * frame_size))

            if self.sampling_strategy == "stratify":

                doc_ids = self._stratify_sample_final(df, sample_n)

            elif self.sampling_strategy == "stratify_even":

                doc_ids = self._stratify_sample_even(df, sample_n)

            elif self.sampling_strategy == "stratify_guaranteed":

                strata_one = self._take_one_per_strata(df, sample_n)
                left_to_sample = sample_n - len(strata_one)
                if left_to_sample > 0:
                    doc_ids = self._stratify_sample_final(df[~df[self.id_col].isin(strata_one)], left_to_sample)
                    doc_ids = list(doc_ids) + list(strata_one)
                else:
                    print("Nothing left to sample, no stratification applied")
                    doc_ids = strata_one

            elif self.sampling_strategy == "stratify_alt":

                doc_ids = self._stratify_sample_alt(df, sample_n)

        if "_stratify_by" in df.columns:
            del df["_stratify_by"]

        print("Sample of %i extracted" % (len(doc_ids)))

        return list(doc_ids)


    def _take_one_per_strata(self, df, sample_n):

        # Number of groups to stratify by must be less than the total sample size
        strata_groups = df.groupby('_stratify_by')[self.id_col].count().count()
        if sample_n > strata_groups:
            print("Sampling one document per strata first ({} strata total)".format(strata_groups))
            one_per = df.groupby('_stratify_by').apply(lambda x: x.sample(1, random_state=self.seed))[self.id_col].values
            return one_per
        else:
            print("There are more strata groups ({}) than things to sample: {}".format(strata_groups, sample_n))
            one_per = df.groupby('_stratify_by').apply(lambda x: x.sample(1, random_state=self.seed))[self.id_col].sample(sample_n).values
            return one_per

    def _stratify_sample_even(self, df, sample_size):

        """
        Sample evenly from each group that's stratified by.
        At least one document per group
        """
        random.seed(self.seed)
        docs_per_strata = int(float(sample_size)/ float(df.groupby('_stratify_by')[self.id_col].count().count()))
        print("Drawing even samples of {} across all stratification groups".format(docs_per_strata))
        doc_ids = []
        for strata in df['_stratify_by'].unique():
            strata_data = df[df['_stratify_by'] == strata]
            doc_ids.extend(list(strata_data.sample(docs_per_strata)[self.id_col].values))
        if len(doc_ids) <  sample_size:
            doc_ids.extend(list(df.sample(sample_size - len(doc_ids))[self.id_col].values))

        return doc_ids

    def _stratify_sample_final(self, df, sample_n):

        """
            Proportional stratification

            Method sourced from : Kish, Leslie. "Survey sampling." (1965).  Chapter 4.

        """
        print("Kish-style stratification")
        # Subset & copy cols that we care about
        data = df.copy()[[self.id_col] + ['_stratify_by']]
        frame_size = data.shape[0]

        # Shuffle the dataframe
        if self.logger:
            self.logger.debug("Random seed: {}".format(self.seed))
        np.random.seed(self.seed)
        if self.logger:
            self.logger.debug("Dataframe before sorting:{}".format(data.head()))
        data.index = np.random.permutation(data.index)
        # Re-sort grouped by strata
        data = data.groupby('_stratify_by').apply(lambda x: x.sort_index())
        data.index = list(range(0, frame_size))
        if self.logger:
            self.logger.debug("Dataframe after shuffle & groupby sorting:{}".format(data.head()))

        skip_interval = float(frame_size) / float(sample_n)

        start_index = np.random.uniform(0, skip_interval) # index to start from
        if self.logger:
            self.logger.info("start index : {}".format(start_index))
        mysample_index = np.round((np.zeros(sample_n) + start_index) + (np.arange(sample_n) * skip_interval))

        # Return the real id column
        mysample_id = data[data.index.isin(mysample_index)][self.id_col].values

        return mysample_id

    def _stratify_sample_alt(self, df, sample_size):
        """
        In stratified sampling, the overall proportion of each group is preserved

        :param df: dataframe to sample from
        :param sample_size: sample size ( int )
        :param stratify_by: column name to stratify by ( or list of columns )
        :param id_col:
        :return: id column for the sample & stats about the sample
        """
        print("New stratification method")
        docs_per_strata = {}
        if '_stratify_by':

            def group_percentages(df, groupcols, id_col=None):

                """
                Returns percentages (summing to 1) of each group out of total
                """

                if not id_col:
                    id_col = df.index

                return pd.pivot_table(
                    df,
                    id_col,
                    groupcols,
                    aggfunc=[len]
                ).apply(
                    lambda x: x / x.sum()
                )['len']

            strata_proportions = group_percentages(df, '_stratify_by', self.id_col)[self.id_col]
            docs_per_strata = (strata_proportions * sample_size).round().to_dict()

            doc_ids = []
            for strata, strata_data in df.groupby('_stratify_by'):
                sample_strata_ids = self._basic_sample(strata_data[self.id_col], docs_per_strata[strata]).values.tolist()
                doc_ids = doc_ids + sample_strata_ids

            print("Stratified sample of %i extracted" % (len(doc_ids)))

        else:
            doc_ids = self._basic_sample(df[self.id_col], sample_size, seed=self.seed).values.tolist()

        return doc_ids

    def _basic_sample(self, df, size):

        if size >= 1:
            try:
                return df.sample(int(size), random_state=self.seed)
            except ValueError:
                return df
        else:
            return df.sample(frac=float(size), random_state=self.seed)
