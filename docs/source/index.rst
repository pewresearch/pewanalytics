.. include:: home.rst


First steps
-----------

Pewanalytics is a package containing many of the text and statistics utilities developed \
at the Pew Research Center over the years. We've seperated things out into a text and a stats \
submodule to text processing tasks including using `topic models to analyze survey open ends <link here>_`
and using `cosine similarity and fuzzy matching to identify duplicates in the
FCC comments on net neutrality <link here>_`.


To install ::

    pip install https://github.com/pewresearch/pewanalytics#egg=pewanalytics

Or install from source ::

    git clone https://github.com/pewresearch/pewanalytics.git
    cd pewanalytics
    python setup.py install


.. note::
    This is a Python3 package. Though it's compatible with Python 2, its dependencies are \
    planning to drop support for earlier versions. We highly recommend you upgrade to Python3.


Tests ::

    pytest




Our examples


******************
Full Documentation
******************

This contains everything you need to know about every function in the source code.


.. toctree::
   :caption: Navigation
   :maxdepth: 2

   Text <text>
   Stats <stats>

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
