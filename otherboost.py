import pandas as pd
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def do_boosting(df, df_name, xgb_df, xgb_df_f):
    tot_rows = df.shape[0]
    train_size = int(tot_rows*0.8)
    train = df[:train_size]
    test = df[train_size:]

    X = train
    X = X.drop(['revenue'], axis=1)
    y = train.revenue.apply(np.log1p)

    X_predict = test
    X_predict = X_predict.drop(['revenue'], axis=1)
    y_predict = test.revenue.apply(np.log1p)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12, shuffle=True)
    params = {'objective': 'reg:squarederror',
              'eta': 0.01,
              'max_depth': 6,
              'min_child_weight': 3,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.50,
              'gamma': 1.45,
              'eval_metric': 'rmse',
              'seed': 12,
              'silent': True
              }
    xgb_data = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_test, y_test), 'valid')]
    xgb_model = xgb.train(params,
                          xgb.DMatrix(X_train, y_train),
                          5000,
                          xgb_data,
                          verbose_eval=200,
                          early_stopping_rounds=200)
    xgb_model_full = xgb.XGBRegressor(objective='reg:squarederror',
                                      eta=0.01,
                                      max_depth=6,
                                      min_child_weight=3,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      colsample_bylevel=0.50,
                                      gamma=1.45,
                                      eval_metric='rmse',
                                      seed=12, n_estimators=2000)
    xgb_model_full.fit(X.values, y)
    f_importance = xgb_model_full.feature_importances_
    xgb_pred = np.expm1(xgb_model.predict(xgb.DMatrix(X_predict), ntree_limit=xgb_model.best_ntree_limit))
    xgb_pred_f = np.expm1(xgb_model_full.predict(X_predict.values))
    if xgb_df.empty:
        xgb_df = pd.concat([xgb_df, pd.DataFrame({'id': test.id, 'revenue_%s' % df_name: xgb_pred})])
        xgb_df_f = pd.concat([xgb_df_f, pd.DataFrame({'id': test.id, 'revenue_%s' % df_name: xgb_pred_f})])
    else:
        xgb_df = pd.merge(xgb_df, pd.DataFrame({'id': test.id, 'revenue_%s' % df_name: xgb_pred}), on='id')
        xgb_df_f = pd.merge(xgb_df_f, pd.DataFrame({'id': test.id, 'revenue_%s' % df_name: xgb_pred_f}), on='id')
    return xgb_df, xgb_df_f, f_importance


def build_diff_df(df):
    temp_df = df
    df_without_cast = temp_df.drop(['cast_popularity', 'top5_popularity', 'num_females', 'is_lead_female'], axis=1)
    df_without_crew = temp_df.drop(['num_crew', 'num_director', 'director_popularity'], axis=1)
    df_sparse = temp_df[['id', 'budget', 'revenue', 'release_year', 'genres_popularity']]
    df_without_budget = temp_df.drop(['budget', 'inflation_budget', 'scaled_budget',
                                      'budget_num_cast_ratio', 'budget_runtime_ratio'], axis=1)
    return [(df, 'all features'), (df_without_cast, 'without cast'),
            (df_without_crew, 'without crew'), (df_sparse, 'sparse features'),
            (df_without_budget, 'no budget')]


def main():
    dataframe = pd.read_csv('datasets/filtered_dataset.csv')
    df_list = build_diff_df(dataframe)
    pred_list = []
    xgb_df = pd.DataFrame()
    xgb_df_f = pd.DataFrame()
    fi_list = []

    for df, df_name in df_list:
        print(df_name)
        xgb_df, xgb_df_f, f_importance = do_boosting(df, df_name, xgb_df, xgb_df_f)
        fi_list.append(f_importance)
        print('--------------------------------------------------------------')

    tot_rows = df.shape[0]
    train_size = int(tot_rows * 0.8)
    test = df[train_size:]

    xgb_df = pd.merge(xgb_df, pd.DataFrame({'id': test.id, 'real_revenue': test.revenue}), on='id')
    xgb_df_f = pd.merge(xgb_df_f, pd.DataFrame({'id': test.id, 'real_revenue': test.revenue}), on='id')
    xgb_df.to_csv('xgbsubmission.csv', index=False)
    xgb_df_f.to_csv('xgbfullsubmission.csv', index=False)
    features_df = pd.DataFrame(fi_list)
    features_df.to_csv('new_feature_importance.csv')

    print('finish')

if __name__ == "__main__":
    main()