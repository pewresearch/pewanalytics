from builtins import str
from builtins import object
import nltk
import spacy
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, ne_chunk
from pewtils import decode_text


class NamedEntityExtractor(object):

    """

    A wrapper around NLTK and SpaCy for named entity extraction. May be expanded to include more libraries in the \
    future.

    :param method: Specify the library to use when extracting methods. Options are 'nltk', 'spacy', 'all'. If \
    'all' is selected, both libraries will be used and the union will be returned. (Default='spacy')
    :type method: str

    Usage::

        from pewanalytics.text.ner import NamedEntityExtractor
        import nltk

        nltk.download("inaugural")
        fileid = nltk.corpus.inaugural.fileids()[0]
        text = nltk.corpus.inaugural.raw(fileid)

        >>> ner = NamedEntityExtractor(method="nltk")
        >>> ner.extract(text)
        {
            'ORGANIZATION': [
                'Parent', 'Invisible Hand', 'Great Author', 'House', 'Constitution', 'Senate',
                'Human Race', 'Representatives'
            ],
            'PERSON': ['Almighty Being'],
            'GPE': ['Heaven', 'United States', 'American']
        }

        >>> ner = NamedEntityExtractor(method="spacy")
        >>> ner.extract(text)
        {
            'ORGANIZATION': ['House of Representatives', 'Senate', 'Parent of the Human Race'],
            'DATE': ['present month', 'every day', '14th day', 'years'],
            'ORDINAL': ['first', 'fifth'],
            'GPE': ['United States'],
            'NORP': ['republican', 'American'],
            'LAW': ['Constitution']
        }

        >>> ner = NamedEntityExtractor(method="all")
        >>> ner.extract(text)
        {
            'ORGANIZATION': [
                'Representatives', 'Great Author', 'House', 'Parent', 'House of Representatives',
                'Parent of the Human Race', 'Invisible Hand', 'Human Race', 'Senate', 'Constitution'
            ],
            'PERSON': ['Almighty Being'],
            'GPE': ['Heaven', 'United States', 'American'],
            'DATE': ['every day', 'present month', '14th day', 'years'],
            'ORDINAL': ['first', 'fifth'],
            'NORP': ['republican', 'American'],
            'LAW': ['Constitution']
        }

    """

    def __init__(self, method="spacy"):

        if method not in ["nltk", "spacy", "all"]:
            raise Exception("Available methods are: 'nltk', 'spacy', 'all'")
        self.method = method

        self.type_map = {
            "ORG": "ORGANIZATION",
            "PER": "PERSON",
            "LOC": "LOCATION",
            "FAC": "FACILITY",
            "VEH": "VEHICLE",
            "WEA": "WEAPON",
            "GSP": "GPE",
        }

    def extract(self, text):

        """
        :param text: a string from which to extract named entities
        :type text: str
        :return: dictionary of entities organized by their category
        :rtype: dict
        """

        try:
            text = str(text)
        except Exception as e:
            text = decode_text(text)

        roots = defaultdict(list)

        if self.method in ["nltk", "all"]:

            try:
                tree = ne_chunk(pos_tag(word_tokenize(text)))
            except LookupError:
                nltk.download("maxent_ne_chunker")
                nltk.download("words")
                tree = ne_chunk(pos_tag(word_tokenize(text)), binary=True)

            for branch in tree:
                if type(branch) is nltk.Tree:
                    leaf = [" ".join(x[0] for x in branch.leaves())]
                    key = self.type_map.get(branch.label(), branch.label())
                    roots[key].extend(leaf)

        if self.method in ["spacy", "all"]:

            # SpaCy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
            for entity in nlp(text).ents:
                entity_text = re.sub(
                    r"^({})\s".format("|".join(stopwords.words("english"))),
                    "",
                    entity.text,
                )
                key = self.type_map.get(entity.label_, entity.label_)
                roots[key].append(entity_text)

        return {self.type_map.get(k, k): list(set(v)) for k, v in roots.items()}
