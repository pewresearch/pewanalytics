****************************
pewanalytics: Text Module
****************************


Some simple things you can do
-----------------------------

Parts of speech ::

    from pewanalytics.text import filter_parts_of_speech
    df['original_text'].values[0]
    filter_parts_of_speech(doc, filter_pos=["NN"])

Tokenize sentences ::

    from pewanalytics.text import SentenceTokenizer
    SentenceTokenizer

Clean text ::

    from pewanalytics.text import TextCleaner
    clean_text = TextCleaner.clean(messy_text,
                                decode_text = True,
                                lemmatize = True,
                                lemmatizer = my_personal_favorite_lemmatizer)


Analyze and classify lists of documents ::

    from pewanalytics.text import TextDataFrame

.. rubric:: Text Modules
``text.<function>``

.. automodule :: pewanalytics.text
  :noindex:
  :no-members:
  :autosummary:
  :autosummary-members:
  :autosummary-inherited-members:


.. rubric:: Text: Dates
``text.dates.<class>``

.. automodule :: pewanalytics.text.dates
  :noindex:
  :no-members:
  :autosummary:
  :autosummary-members:


.. rubric:: Text: Named Entity Recognition
``text.ner.<class>``

.. automodule :: pewanalytics.text.ner
  :noindex:
  :no-members:
  :autosummary:
  :autosummary-members:


Text Methods
--------------

.. warning ::
    Running these functions requires text contain no special characters

.. automodule :: pewanalytics.text
    :members:
    :show-inheritance:

Dates
-----

.. automodule :: pewanalytics.text.dates
    :members:
    :show-inheritance:


Named Entity Recognition
------------------------

A light-weight wrapper using `NLTK's Named Entity Recognition
model <https://www.nltk.org/book/ch07.html>`_.

To use ::

  from pewanalytics.text.ner import NamedEntityExtractor
  extractor = NamedEntityExtractor(messy_text)
  roots = extractor.extract()

.. automodule :: pewanalytics.text.ner
    :members:

.. toctree::
   :caption: Navigation
   :maxdepth: 2

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
