import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec, KeyedVectors
from datetime import datetime



from loading_script import get_movies_db, dictioning_list, dictioning_column, init_genres_vocab, geners_vocab_list, \
    get_number_list, turn_string_list_to_int, init_cast_vocab
TRAIN_CSV_PATH = './datasets/train.csv'
TEST_CSV_PATH = './datasets/test.csv'
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
    clf = LogisticRegression(random_state=random_state)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

def w2v(column):
    path = get_tmpfile("word2vec.model")
    model = Word2Vec(column, size=100, window=5, min_count=1, workers=4)  # movieDB[movie]
    model.save("word2vec.model")
    model = Word2Vec.load("word2vec.model")
    trained = model.train([["Family", "Comedy"]], total_examples=1, epochs=1)
    path = get_tmpfile("wordvectors.kv")
    model.wv.save(path)
    wv = KeyedVectors.load(path, mmap='r')
    vector = wv['Comedy']
    wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)
    wv_from_bin = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True)
    return wv

'''
def listing_word_vectors(f_col, f_vector, f_index2word):
    worderd_column = []
    for movie_id, movie in enumerate(f_col):
        worderd_column.append([])
        for item in movie:
            for word_id, word in enumerate(f_index2word):
                if item == word:
                    worderd_column[movie_id].append(f_vector[word_id])
                else:
                    worderd_column[movie_id].append(np.zeros(len(f_vector[word_id])))
    return worderd_column
'''




# decision tree
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

def test1(dataset):
    GENRES_VOCAB = geners_vocab_list(dataset)
    data = w2v(GENRES_VOCAB)
    f_vectors, index2word = data.vectors, data.index2word
    worded_feature = listing_word_vectors(GENRES_VOCAB ,f_vectors, index2word)
    revenue = dataset["revenue"]
    target = revenue_list(revenue)
    regressor = DecisionTreeRegressor(random_state=0)
    res = cross_val_score(regressor, worded_feature, target, cv=10)
    regressor.fit(data, target)
    print("end of test")


from sklearn.model_selection import KFold
import xgboost as xgb
random_seed = 2019
k = 10
np.random.seed(random_seed)
def xgb_model(trn_x, trn_y, val_x, val_y, test, verbose):
    params = {'objective': 'reg:linear',
              'eta': 0.01,
              'max_depth': 6,
              'subsample': 0.6,
              'colsample_bytree': 0.7,
              'eval_metric': 'rmse',
              'seed': random_seed,
              'silent': True,
              }

    record = dict()
    model = xgb.train(params
                      , xgb.DMatrix(trn_x, trn_y)
                      , 100000
                      , [(xgb.DMatrix(trn_x, trn_y), 'train'), (xgb.DMatrix(val_x, val_y), 'valid')]
                      , verbose_eval=verbose
                      , early_stopping_rounds=500
                      , callbacks=[xgb.callback.record_evaluation(record)])
    best_idx = np.argmin(np.array(record['valid']['rmse']))

    val_pred = model.predict(xgb.DMatrix(val_x), ntree_limit=model.best_ntree_limit)
    test_pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)

    return {'val': val_pred, 'test': test_pred, 'error': record['valid']['rmse'][best_idx],
            'importance': [i for k, i in model.get_score().items()]}


def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d


def main():
    #""" xgboost

    train = get_movies_db(TRAIN_CSV_PATH)
    TR_CAST_VOCAB = init_cast_vocab(train)
    tr_cast = dictioning_column(train["cast"])
    train = train
    train['genres'] = train['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(
        lambda x: ','.join(map(str, x)))
    p_train = train ['revenue']
    cast_int_list = turn_string_list_to_int(tr_cast, TR_CAST_VOCAB)
    train.insert(1, "INT LIST", cast_int_list)
    train = train[['budget', 'popularity', 'runtime', 'INT LIST']]


    ###############################################33 test
    test = get_movies_db(TEST_CSV_PATH)
    test = test
    TS_CAST_VOCAB = init_cast_vocab(test)
    ts_cast = dictioning_column(test["cast"])
    test['genres'] = test['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(
        lambda x: ','.join(map(str, x)))
    p_test = ['revenue']

    cast_int_list = turn_string_list_to_int(ts_cast, TS_CAST_VOCAB)
    test.insert(1, "INT LIST", cast_int_list)
    test = test[['budget', 'popularity', 'runtime', 'INT LIST']]
    random_seed = 2019
    k = 10
    fold = list(KFold(k, shuffle=True, random_state=random_seed).split(train))
    np.random.seed(random_seed)
    result_dict = dict()
    val_pred = np.zeros(train.shape[0])
    test_pred = np.zeros(test.shape[0])
    final_err = 0
    verbose = False
    for i, (trn, val) in enumerate(fold):
        print(i + 1, "fold.    RMSE")
        y = p_train
        trn_x = train.loc[trn, :]
        trn_y = y[trn]
        val_x = train.loc[val, :]
        val_y = y[val]

        fold_val_pred = []
        fold_test_pred = []
        fold_err = []

        start = datetime.now()
        result = xgb_model(trn_x, trn_y, val_x, val_y, test, verbose=False)
        fold_val_pred.append(result['val']*0.2)
        fold_test_pred.append(result['test']*0.2)
        fold_err.append(result['error'])
        print("xgb model.", "{0:.5f}".format(result['error']), '(' + str(int((datetime.now()-start).seconds/60)) + 'm)')
        test_pred += np.mean(np.array(fold_test_pred), axis = 0) / k
        final_err += (sum(fold_err) / len(fold_err)) / k

        print("---------------------------")
        print("avg   err.", "{0:.5f}".format(sum(fold_err) / len(fold_err)))
        print("blend err.", "{0:.5f}".format(np.sqrt(np.mean((np.mean(np.array(fold_val_pred), axis = 0) - val_y)**2))))

        print('')

    print("final avg   err.", final_err)
    print("final blend err.", np.sqrt(np.mean((val_pred - y)**2)))
if __name__ == "__main__":
    main()