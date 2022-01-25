*************************************
pewanalytics.stats: Statistical Tools
*************************************

In the :py:mod:`pewanalytics.stats` module, you'll find a variety of statistical utilities for \
weighting, clustering, dimensionality reduction, and inter-rater reliability.

Clustering
-----------------------------

The :py:mod:`pewanalytics.stats.clustering` submodule contains several functions for extracting \
clusters from your data.

.. automodule :: pewanalytics.stats.clustering
  :autosummary:
  :autosummary-members:
  :members:

Dimensionality Reduction
--------------------------------------------

The :py:mod:`pewanalytics.stats.dimensionality_reduction` submodule contains functions for \
collapsing your data into underlying dimensions using methods like PCA and correspondence \
analysis.

.. automodule :: pewanalytics.stats.dimensionality_reduction
  :autosummary:
  :autosummary-members:
  :members:

Inter-Rater Reliability
-----------------------

The :py:mod:`pewanalytics.stats.irr` submodule contains functions for computing measures of \
inter-rater reliability and model performance, including Cohen's Kappa, Krippendorf's Alpha, \
precision, recall, and much more.

.. automodule :: pewanalytics.stats.irr
  :autosummary:
  :autosummary-members:
  :members:

Mutual Information
------------------------------

The :py:mod:`pewanalytics.stats.mutual_info` submodule provides a function for extracting \
pointwise mutual information for features in your data based on a binary split into two \
classes. This can be a great method for identifying features that are most distinctive \
of one group versus another.

.. automodule :: pewanalytics.stats.mutual_info
  :autosummary:
  :autosummary-members:
  :members:

Sampling
---------------------------

The :py:mod:`pewanalytics.stats.sampling` submodule contains utilities for extracting and \
weighting samples based on a known sampling frame.

.. automodule :: pewanalytics.stats.sampling
  :autosummary:
  :autosummary-members:
  :members:
