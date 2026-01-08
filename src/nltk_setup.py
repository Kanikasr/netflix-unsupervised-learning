import nltk
import logging


def setup_nltk():
    # Silence NLTK downloader logging
    logging.getLogger('nltk').setLevel(logging.ERROR)

    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet'
    }

    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)
