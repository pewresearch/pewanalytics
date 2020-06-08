Pew Analytics
===================================================================


Pew Analytics is a package containing many of the text and statistics utilities developed \
at the Pew Research Center over the years. Many of our research projects involve routine tasks specifically related \
to processing and analyzing data: things like cleaning up text documents, de-duplicating records, and looking for \
hidden clusters and groups. This package contains a collection of tools designed to make these tasks easier.

.. toctree::
   :maxdepth: 1
   :caption: Table of Contents:

   Statistical Tools <stats>
   Text Tools <text>
   Examples <examples>

Installation
---------------

To install, you can use ``pip``: ::

    pip install git+https://github.com/pewresearch/pewanalytics#egg=pewanalytics

Or you can install from source: ::

    git clone https://github.com/pewresearch/pewanalytics.git
    cd pewanalytics
    python setup.py install


.. note::
    This is a Python 3 package. Though it is compatible with Python 2, many of its dependencies are \
    planning to drop support for earlier versions if they haven't already. We highly recommend \
    you upgrade to Python 3.
