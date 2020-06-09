**************
Examples
**************


Sampling
-----------------------------------------------------

The :py:mod:`pewanalytics.stats.sampling` module has several useful tools for extracting samples and \
computing sampling weights. Given a sampling frame stored in a :py:class:`pandas.DataFrame`, you can draw \
a sample using a variety of different sampling methods, and then compute sampling weights for any \
combination of one or more binary variables.

.. code-block:: python

    from pewanalytics.stats.sampling import SampleExtractor, compute_sample_weights_from_frame

    import nltk
    import pandas as pd
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.feature_extraction.text import TfidfVectorizer

    nltk.download("inaugural")
    frame = pd.DataFrame([
        {"speech": fileid, "text": nltk.corpus.inaugural.raw(fileid)} for fileid in nltk.corpus.inaugural.fileids()
    ])

    # Let's set some flags what we'll use in sampling
    frame['economy'] = frame['text'].str.contains("economy").astype(int)
    frame['health'] = frame['text'].str.contains("health").astype(int)
    frame['immigration'] = frame['text'].str.contains("immigration").astype(int)
    frame['education'] = frame['text'].str.contains("education").astype(int)

    # Now we can grab a sample of speeches, stratifying by these variables
    # This will ensure that our sample will contain speeches that mention each term
    stratification_variables = ["economy", "health", "immigration", "education"]
    extractor = SampleExtractor(frame, "speech")
    sample_index = extractor.extract(
        10,
        sampling_strategy="stratify",
        stratify_by=stratification_variables
    )
    sample = frame[frame['speech'].isin(sample_index)]
    >>> sample[stratification_variables].sum()
    economy        5
    health         3
    immigration    1
    education      4
    dtype: int64

    # Now we can get sampling weights to adjust our sample back to the population based on our stratification variables
    sample['weight'] = compute_sample_weights_from_frame(
        frame,
        sample,
        stratification_variables
    )
    >>> sample
                    speech                                               text  \
    2       1797-Adams.txt  When it was first perceived, in early times, t...
    10    1829-Jackson.txt  Fellow citizens, about to undertake the arduou...
    17   1857-Buchanan.txt  Fellow citizens, I appear before you this day ...
    18    1861-Lincoln.txt  Fellow-Citizens of the United States: In compl...
    27   1897-McKinley.txt  Fellow citizens, In obedience to the will of t...
    37  1937-Roosevelt.txt  When four years ago we met to inaugurate a Pre...
    46      1973-Nixon.txt  Mr. Vice President, Mr. Speaker, Mr. Chief Jus...
    49     1985-Reagan.txt  Senator Mathias, Chief Justice Burger, Vice Pr...
    51    1993-Clinton.txt  My fellow citizens, today we celebrate the mys...
    52    1997-Clinton.txt  My fellow citizens: At this last presidential ...

        economy  health  immigration  education  count    weight
    2         0       0            0          0      1  0.919540
    10        1       0            0          0      1  0.948276
    17        0       0            0          0      1  0.919540
    18        0       0            0          0      1  0.919540
    27        1       0            1          1      1  0.689655
    37        0       0            0          1      1  0.862069
    46        0       1            0          1      1  0.172414
    49        1       0            0          0      1  0.948276
    51        1       1            0          0      1  1.206897
    52        1       1            0          1      1  0.862069

Inter-rater reliability
-----------------------------------------------------

The :py:mod:`pewanalytics.stats.irr` module has several useful functions for computing a wide variety \
of inter-rater reliability and model performance metrics. The :py:func:`pewanalytics.stats.irr.compute_scores` \
function provides a one-stop shop for assessing agreement between two classifiers - whether you're comparing coders \
or machine learning models.

.. code-block:: python

    from pewanalytics.stats.irr import compute_scores

    import pandas as pd

    # Let's create a DataFrame with some fake classification decisions. We'll make one with two coders,
    # three documents, and three possible codes
    df = pd.DataFrame([
        {"coder": "coder1", "document": 1, "code": "2"},
        {"coder": "coder2", "document": 1, "code": "2"},
        {"coder": "coder1", "document": 2, "code": "1"},
        {"coder": "coder2", "document": 2, "code": "2"},
        {"coder": "coder1", "document": 3, "code": "0"},
        {"coder": "coder2", "document": 3, "code": "0"},
    ])

    # To get overall average performance metrics, we can pass the DataFrame in like so:
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

    # And if we want to get the scores for a specific code/label (comparing it against all other possible codes)
    # Then we can specify it using the `pos_label` keyword argument:

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


Cleaning text
-----------------------------------------------------

When working with text data, pre-processing is an essential first task. The :py:mod:`pewanalytics.text` module \
contains a wide range of tools for working with text data, among them the :py:class:`pewanalytics.text.TextCleaner` \
class that provides a wide range of pre-processing options for cleaning your text.

.. code-block:: python

    from pewanalytics.text import TextCleaner

    text = """
        <body>
        Here's some example text.</br>It isn't a great example, but it'll do.
        Of course, there are plenty of other examples we could use though.
        http://example.com
        </body>
    """

    >>> cleaner = TextCleaner(process_method="stem")
    >>> cleaner.clean(text)
    'exampl is_not great exampl cours plenti exampl could use though'

    >>> cleaner = TextCleaner(process_method="stem", stopwords=["my_custom_stopword"], strip_html=True)
    >>> cleaner.clean(text)
    'here some exampl is_not great exampl but will cours there are plenti other exampl could use though'

    >>> cleaner = TextCleaner(process_method="lemmatize", strip_html=True)
    >>> cleaner.clean(text)
    'example is_not great example course plenty example could use though'

    >>> cleaner = TextCleaner(process_method="lemmatize", remove_urls=False, strip_html=True)
    >>> cleaner.clean(text)
    'example text is_not great example course plenty example could use though http example com'

    >>> cleaner = TextCleaner(process_method="stem", strip_html=False)
    >>> cleaner.clean(text)
    'example text is_not great example course plenty example could use though http example com'

    >>> cleaner = TextCleaner(process_method="stem", filter_pos=["JJ"], strip_html=True)
    >>> cleaner.clean(text)
    'great though'

The TextDataFrame class
-----------------------------------------------------

In some of the following examples, we'll be making use of the :py:class:`pewanalytics.text.TextDataFrame` class, \
which provides a variety of useful functions for working with a Pandas DataFrame that contains a column \
of text that you want to analyze. To set up a ``TextDataFrame``, you just need to pass a DataFrame and \
specify the name of the column that contains the text. The ``TextDataFrame`` will automatically convert \
your corpus into a TF-IDF representation; you can pass additional keyword arguments to control this \
vectorization process, which get forwarded to a Scikit-Learn ``TfidfVectorizer`` class. In the following examples, \
we'll be using a ``TextDataFrame`` containing inaugural speeches:

.. code-block:: python

    from pewanalytics.text import TextDataFrame
    import pandas as pd
    import nltk

    nltk.download("inaugural")
    df = pd.DataFrame([
        {"speech": fileid, "text": nltk.corpus.inaugural.raw(fileid)} for fileid in nltk.corpus.inaugural.fileids()
    ])
    # Let's remove new line characters so we can print the output in the docstrings
    df['text'] = df['text'].str.replace("\n", " ")

    # And now let's create some additional variables to group our data
    df['year'] = df['speech'].map(lambda x: int(x.split("-")[0]))
    df['21st_century'] = df['year'].map(lambda x: 1 if x >= 2000 else 0)

    # And we'll also create some artificial duplicates in the dataset
    df = df.append(df.tail(2)).reset_index()

    # We'll use this TextDataFrame in a variety of examples below
    tdf = TextDataFrame(df, "text", stop_words="english", ngram_range=(1, 2))


Finding repeating fragments
-----------------------------------------------------

When working with text, sometimes it can be useful to identify repeating segments of text that \
occur in multiple documents. For example, you might be interested in identifying common phraess, \
or perhaps you want to look for common boilerplate text that you want to clear out in order to \
facilitate more accurate document comparison. In these cases, Pew Analytics provides several functions \
that can help.

The ``TextOverlapExtractor`` can identify overlaps between two pieces of text:

.. code-block:: python

    from pewanalytics.text import TextOverlapExtractor

    text1 = "This is a sentence. This is another sentence. And a third sentence. And yet a fourth sentence."
    text2 = "This is a different sentence. This is another sentence. And a third sentence. But the fourth \
    sentence is different too."

    >>> extractor = TextOverlapExtractor()

    >>> extractor.get_text_overlaps(text1, text2, min_length=10, tokenize=False)
    [' sentence. This is another sentence. And a third sentence. ', ' fourth sentence']

    >>> extractor.get_text_overlaps(text1, text2, min_length=10, tokenize=True)
    ['This is another sentence.', 'And a third sentence.']

If you want to apply this function at scale, you can make use of the ``TextDataFrame`` to search for \
repeating fragments of text that occur across a large corpus. This function uses the ``TextOverlapExtractor``, \
which tokenizes your text into complete sentences by default. In our example, there aren't any unique sentences \
that recur, but we can disable tokenization to get raw overlapping segments of text like so:

.. code-block:: python

    >>> tdf.extract_corpus_fragments(scan_top_n_matches_per_doc=20, min_fragment_length=25, tokenize=False)
    ['s. Equal and exact justice ',
     'd by the General Government',
     ' of the American people, ',
     'ent of the United States ',
     ' the office of President of the United States ',
     ' preserve, protect, and defend the Constitution of the United States."  ',
     ' to "preserve, protect, and defend',
     ' of the United States are ',
     'e of my countrymen I am about to ',
     'Vice President, Mr. Chief Justice, ',
     ' 200th anniversary as a nation',
     ', and my fellow citizens: ',
     'e United States of America']

Finding duplicates
-----------------------------------------------------

Text corpora also often contain duplicates that we want to remove prior to analysis. \
To efficiently identify these duplicates, the ``TextDataFrame`` provides a two-step function \
that uses TF-IDF to identify potential duplicate pairs, which are then filtered down by using \
more precise Levenshtein ratios:

.. code-block:: python

    >>> tdf.find_duplicates()
    [           speech                                               text  year
    56  2013-Obama.txt  Thank you. Thank you so much.    Vice Presiden...  2013
    56  2013-Obama.txt  Thank you. Thank you so much.    Vice Presiden...  2013

        21st_century
    56             1
    56             1  ,
                speech                                               text  year
    57  2017-Trump.txt  Chief Justice Roberts, President Carter, Presi...  2017
    57  2017-Trump.txt  Chief Justice Roberts, President Carter, Presi...  2017

        21st_century
    57             1
    57             1  ]


Mutual information
-----------------------------------------------------

Pointwise mutual information can be an enormously useful tool for identifying words and phrases \
that distinguish one group of documents from another. The :py:mod:`pewanalytics.stats.mutual_info` module \
contains a ``mutual_info`` function for computing mutual information along with a variety of other \
ratios that identify features that distinguish between two different sets of observations. \
While you can run this function on any set of features, it's particularly informative when \
working with text data. Accordingly, the ``TextDataFrame`` has a shortcut function that allows you \
to easily run mutual information on your corpus. In this example, we can find the phrases that \
most distinguish 21st century inaugural speeches from those given in prior years:

.. code-block:: python

    results = tdf.mutual_info("21st_century")

    # Pointwise mutual information for our positive class is stored in the "M1" column
    >>> results.sort_values("MI1", ascending=False).index[:25]
    Index(['journey complete', 'jobs', 'make america', 've', 'obama', 'workers',
           'xand', 'states america', 'america best', 'debates', 'clinton',
           'president clinton', 'trillions', 'stops right', 'transferring',
           'president obama', 'stops', 'protected protected', 'transferring power',
           'nation capital', 'american workers', 'politicians', 'people believe',
           'borders', 'victories'],
           dtype='object')


Topic modeling
-----------------------------------------------------

Just like the ``TextDataFrame``, ``pewanalytics`` also provides a wrapper class for training \
a variety of different topic models. The :py:class:`pewanalytics.text.topics.TopicModel` class accepts \
a Pandas DataFrame and the name of a text column, and allows you to train and apply \
Gensim, Scikit-Learn, and Corex topic models using a standardized interface:

.. code-block:: python

    from pewanalytics.text.topics import TopicModel
    import pandas as pd
    import nltk

    nltk.download("inaugural")
    df = pd.DataFrame([
        {"speech": fileid, "text": nltk.corpus.inaugural.raw(fileid)} for fileid in nltk.corpus.inaugural.fileids()
    ])
    >>> model = TopicModel(df, "text", "sklearn_nfm", num_topics=5, min_df=25, max_df=.5, use_tfidf=False)
    Initialized sklearn_nmf topic model with 3285 features
    1600 training documents, 400 testing documents

    >>> model.fit()

    >>> model.print_topics()
    0: bad, really, know, don, plot, people, scene, movies, action, scenes
    1: star, trek, star trek, effects, wars, star wars, special, special effects, movies, series
    2: jackie, films, chan, jackie chan, hong, master, drunken, action, tarantino, brown
    3: life, man, best, characters, new, love, world, little, does, great
    4: alien, series, aliens, characters, films, television, files, quite, mars, action

    >>> doc_topics = model.get_document_topics(df)

    >>> doc_topics
           topic_0   topic_1   topic_2   topic_3   topic_4
    0     0.723439  0.000000  0.000000  0.000000  0.000000
    1     0.289801  0.050055  0.000000  0.000000  0.000000
    2     0.375149  0.000000  0.030691  0.059088  0.143679
    3     0.152961  0.010386  0.000000  0.121412  0.015865
    4     0.294005  0.100426  0.000000  0.137630  0.051241
    ...        ...       ...       ...       ...       ...
    1995  0.480983  0.070431  0.135178  0.256951  0.000000
    1996  0.139986  0.000000  0.000000  0.107430  0.000000
    1997  0.141545  0.005990  0.081986  0.387859  0.057025
    1998  0.029228  0.023342  0.043713  0.280877  0.107551
    1999  0.044863  0.000000  0.000000  0.718677  0.000000