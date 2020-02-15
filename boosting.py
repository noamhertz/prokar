import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from gensim.test.utils import common_texts, get_tmpfile, datapath
from gensim.models import Word2Vec, KeyedVectors
from datetime import datetime



from loading_script import get_movies_db, dictioning_list, dictioning_column, init_genres_vocab, genres_vocab_list, get_number_list
filename = os.path.join('.', '/datasets/filtered_dataset.csv')
TRAIN_CSV_PATH = './datasets/filtered_dataset.csv'


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
    import pdb
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


# decision tree
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

def test1(dataset):
    GENRES_VOCAB = genres_vocab_list(dataset)
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

def do_xgb(df):
    number_of_rows = len(df)
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=0.2)
    num_train = int(0.8 * number_of_rows)
    rand_gen = np.random.RandomState(0)
    NORMAL = 100000000
    p_train = train['revenue'].values/NORMAL
    test['revenue'] = test['revenue'].values/NORMAL
    # features_train = train.loc[:, df.columns != 'revenue']
    ###############################################33 test
    # p_test = test['revenue']
    # features_test = test.loc[:, df.columns != 'revenue']
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
        ######################## real
        fold_val_pred = []
        fold_test_pred = []
        fold_err = []

        start = datetime.now()
        result = xgb_model(trn_x, trn_y, val_x, val_y, test, verbose=False)
        fold_val_pred.append(result['val'] * 0.2)
        fold_test_pred.append(result['test'] * 0.2)
        fold_err.append(result['error'])
        print("xgb model.", "{0:.5f}".format(result['error']),
              '(' + str(int((datetime.now() - start).seconds / 60)) + 'm)')
        test_pred += np.mean(np.array(fold_test_pred), axis=0) / k
        final_err += (sum(fold_err) / len(fold_err)) / k

        print("---------------------------")
        print("avg   err.", "{0:.5f}".format(sum(fold_err) / len(fold_err)))
        print("blend err.", "{0:.5f}".format(np.sqrt(np.mean((np.mean(np.array(fold_val_pred), axis=0) - val_y) ** 2))))

        print('')

    print("final avg   err.", final_err)
    print("final blend err.", np.sqrt(np.mean((val_pred - y) ** 2)))


def build_diff_df(df):
    temp_df = df
    df_without_cast = temp_df.drop(['cast_popularity', 'top5_popularity', 'num_females', 'is_lead_female'], axis=1)
    df_without_crew = temp_df.drop(['num_crew', 'num_director', 'director_popularity'], axis=1)
    df_sparse = temp_df[['id', 'budget', 'revenue', 'release_year', 'genres_popularity']]
    df_without_budget = temp_df.drop(['budget'], axis=1)
    return [(df, 'all features'), (df_without_cast, 'without cast'),
            (df_without_crew, 'without crew'), (df_sparse, 'sparse features'), (df_without_budget, 'no budget')]

def main():
    #""" xgboost
    ###
    import sys
    orig_stdout = sys.stdout
    dataframe = get_movies_db(TRAIN_CSV_PATH)
    df_list = build_diff_df(dataframe)
    #sys.stdout = open("results.txt", "w")
    for df, df_name in df_list:
        print(df_name)
        do_xgb(df)

if __name__ == "__main__":
    main()