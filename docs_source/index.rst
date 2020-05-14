Getting Started
===================================================================


Pewanalytics is a package containing many of the text and statistics utilities developed \
at the Pew Research Center over the years. We've seperated things out into a text and a stats \
submodule to text processing tasks including using `topic models to analyze survey open ends <link here>_`
and using `cosine similarity and fuzzy matching to identify duplicates in the
FCC comments on net neutrality <link here>_`.

.. toctree::
   :maxdepth: 1
   :caption: Table of Contents:

   Statistical Tools <stats>
   Text Tools <text>

Installation
---------------

To install, you can use PyPI: ::

    pip install pewanalytics

Or you can install from source: ::

    git clone https://github.com/pewresearch/pewanalytics.git
    cd pewanalytics
    python setup.py install


.. note::
    This is a Python 3 package. Though it is compatible with Python 2, many of its dependencies are \
    planning to drop support for earlier versions if they haven't already. We highly recommend \
    you upgrade to Python 3.
