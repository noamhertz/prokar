import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from loading_script import get_movies_db, dictioning_list, dictioning_column, init_genres_vocab, geners_vocab_list

GENRE_THRESH = 5

def build_sets(df):
    x = df[['director', 'genres']].values
    y = df['Revenue'].values
    number_of_rows = len(df)
    num_train = int(0.8 * number_of_rows)

    rand_gen = np.random.RandomState(0)
    shuffled_indices = rand_gen.permutation(np.arange(len(x)))

    x_train = x[shuffled_indices[:num_train]]
    y_train = y[shuffled_indices[:num_train]]
    x_test = x[shuffled_indices[num_train:]]
    y_test = y[shuffled_indices[num_train:]]
    # pre-process - standartization
    import pdb;
    pdb.set_trace()
    return x_train, y_train, x_test, y_test


def logistic_reg(df):
    x_train, y_train, x_test, y_test = build_sets(df)
    import pdb;
    pdb.set_trace()
    random_state = 38
    clf = LogisticRegression(random_state=random_state)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

def test(dataset):
    GENRES_VOCAB = geners_vocab_list(dataset)

    path = get_tmpfile("word2vec.model")
    model = Word2Vec(GENRES_VOCAB, size=100, window=5, min_count=1, workers=4)
    ct = common_texts
    model.save("word2vec.model")
    model = Word2Vec.load("word2vec.model")
    trained = model.train([["hello", "world"]], total_examples=1, epochs=1)
    vector0 = model.wv['Family']
    vector1 = model.wv['Horror']
    print('end of test')


def main():
    dataset = get_movies_db()
    test(dataset)
    print("end")

if __name__ == "__main__":
    main()