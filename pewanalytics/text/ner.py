from builtins import str
from builtins import object
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from pewtils import decode_text


class NamedEntityExtractor(object):
    def __init__(self):

        """
        A wrapper around NLTK named entity extraction. May be expanded in the future to include NER models from other
        packages like SpaCy.
        """

        pass

    def extract(self, text):

        """
        :param text: a string from which to extract named entities
        :return: dictionary of entities organized by their category
        """

        try:
            text = str(text)
        except Exception as e:
            text = decode_text(text)

        try:
            tree = ne_chunk(pos_tag(word_tokenize(text)))
        except LookupError:
            nltk.download("maxent_ne_chunker")
            nltk.download("words")
            tree = ne_chunk(pos_tag(word_tokenize(text)))

        roots = {}
        for branch in tree:
            if type(branch) is nltk.Tree:
                try:
                    leaf = [" ".join(x[0] for x in branch.leaves())]
                    if branch.label() in list(roots.keys()):
                        roots[branch.label()] += leaf
                    else:
                        roots[branch.label()] = leaf
                except Exception as e:
                    pass

        return roots

class Cooccurance:

    """
    :param text:
    """

    def __init__(self, text):
        try:
            text = str(text)
        except Exceptionas e:
            text = decode_text_brutally(text)

        self.text = text
        self.sparse_matrix = defaultdict(lambda: defaultdict(lambda: 0))
        self.dense_matrix = numpy.zeros((lexicon_size, lexicon_size))

    def sparse_matrix(self):
        text = self.text
        for sent in sent_tokenize(text):
            words = word_tokenize(sent)
            for word1 in words:
                for word2 in words:
                    sparse_matrix[word1][word2]+=1
        return sparse_matrix

    def mod_hash(x, m):
        return hash(x) % m

    def dense():
        for k in sparse_matrix.iterkeys():
            for k2 in sparse_matrix[k].iterkeys():
                dense_matrix[mod_hash(k, lexicon_size)][mod_hash(k2, lexicon_size)] = \
                    sparse_matrix[k][k2]

        return dense_matrix
