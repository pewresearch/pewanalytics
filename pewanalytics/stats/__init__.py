import pandas as pd

"""
.. _stats:

.. tip: Insert tip

.. autosummary::
    :toctree: _autosummary
    :template: clean.rst

    clustering
    irr
    dimensionality_reduction
    mutual_info
    sampling
"""

def compute_boolean_column_proportions(df, columns):

    """
    :param df: data frame
    :param columns: list of strings 
    :return: proportions, list of bools 
    """

    proportions = []
    for term in columns:
        try: proportions.append((term, df[term].value_counts(normalize=True)[True]))
        except KeyError: proportions.append((term, 0.0))
    proportions = pd.DataFrame(proportions)
    proportions.index = proportions[0]
    proportions = proportions[1]

    return proportions

def weighted_quantiles(series, percentile):

    #TODO this appears to be from stackoverflow, investigate further 

    """
    :param series: Pandas Series 
    :param percentile: float 
    """

    xsort = series.sort_values(series.columns[0])
    xsort['index'] = range(len(series))
    p = percentile * series[series.columns[1]].sum()
    pop = float(xsort[xsort.columns[1]][xsort['index']==0])
    i = 0
    while pop < p:
        pop = pop + float(xsort[xsort.columns[1]][xsort['index']==i+1])
        i = i + 1
    return xsort[xsort.columns[0]][xsort['index']==i]


def compute_weighted_summary_stats(df, columns, weight_col):
    """
    :param df: data frame
    :param column: list of strings
    :param weight_col: string
    :return: dict of column and (average and std)
    """
    output = dict()
    for column in columns:
        average = numpy.average(df[column], weights=weights)
        variance = numpy.average((df[column]-average)**2, weights=weights)
        output[column: (average, math.sqrt(variance))]
    return output
