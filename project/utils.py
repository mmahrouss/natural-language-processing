import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    starspace_embeddings = {}
    with open(embeddings_path, 'r') as f:
        for line in f:
            word, *embeddings = line.split('\t')
            starspace_embeddings[word] =\
             np.asarray(embeddings).astype(np.float32)
    return starspace_embeddings, len(embeddings)


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings.
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    result = np.zeros(dim)
    count = 0
    for token in question.split():
        if token in embeddings: 
            result += embeddings[token]
            count +=1
    if count:
        result  = result/count
    return result


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
