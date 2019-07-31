import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier


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
