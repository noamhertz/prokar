import pandas as pd, numpy as np, tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt

tf.compat.v1.disable_eager_execution()

factor = 15.19*3

# Predicts time-series values using fully-convolutional neural network.
# Args: (train_set, test_set) - DataFrames with shingled time-series. (train_labels, test_labels) - DataFrames with the next value of the time series.
#       step - Training rate. max_epochs - Number of training epochs. batch_size - Training batch size. activation - Tensorflow activation function. reg_coeff - Regularization coefficient.
#       tolerance - Amount of consequent epochs in which test loss can increase without stopping the training phase. is_initial - Whether the graph should be restored.
# Returns: Array of the predicted values.
def regression(train_set, test_set, train_labels, test_labels, step=0.001, max_epochs=100, batch_size=1):
    sample_len = len(train_set.columns)
    W0, b0 = tf.keras.backend.random_normal_variable(shape=[sample_len, sample_len], mean=0.1, scale=1), tf.keras.backend.random_normal_variable(shape=[sample_len], mean=0.1, scale=1)
  #  W1, b1 = tf.keras.backend.random_normal_variable(shape=[sample_len, int(sample_len/2)], mean=0.1, scale=1), tf.keras.backend.random_normal_variable(shape=[int(sample_len/2)], mean=0.1, scale=1)
    W2, b2 = tf.keras.backend.random_normal_variable(shape=[sample_len, 1], mean=0.1, scale=1), tf.keras.backend.random_normal_variable(shape=[1], mean=0.1, scale=1)
    x, y = tf.compat.v1.placeholder(tf.float32, [None, sample_len]), tf.compat.v1.placeholder(tf.float32, [None])
    x0 = tf.nn.sigmoid(tf.matmul(x, W0) + b0)
   # x1 = tf.nn.sigmoid(tf.matmul(x0, W2) + b2)
    x2 = tf.matmul(x0, W2) + b2

    loss = tf.reduce_mean(tf.square(x2 - y))
    opt = tf.compat.v1.train.RMSPropOptimizer(step)
    train = opt.minimize(loss, var_list=[W0, b0, W2, b2])
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:

        sess.run(init)
        for epoch in range(max_epochs):
            for batch in range(int(len(train_set.index) / batch_size)):
                sess.run(train, feed_dict={x: train_set.iloc[batch*batch_size:batch*batch_size+batch_size], y: train_labels.iloc[batch*batch_size:batch*batch_size+batch_size]})
            if epoch % 20 == 0:
                print("epoch: %g, train loss: %g" % (epoch, factor*loss.eval(feed_dict={x: train_set, y: train_labels})))


        print("Training finished and saved. Calculating results...")
        pred = np.reshape(x2.eval(feed_dict={x: test_set, y: test_labels}), len(test_labels.index))
        score = mean_squared_error(pred, np.reshape(test_labels.values, len(test_labels.index)))
        print("Done. Averaged test loss: %f" % score)
    return pred

FD_PATH = './datasets/filtered_dataset.csv'
#FD_PATH = './datasets/over10mil.csv'

def main():
    pd.set_option("display.max_colwidth", 10000)
    dataset = pd.read_csv(FD_PATH).fillna(0)
    max_revenue = dataset['revenue'].max()
    for col in dataset.columns:
        if col == 'revenue':
            dataset[col] = dataset[col] / max_revenue
        else:
            dataset[col] = (dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min())


    train = dataset.sample(frac=0.7)
    df = pd.merge(dataset,train, how='outer',indicator=True)
    test = (df.loc[df._merge == 'left_only']).drop(columns='_merge')

    train_label = train['revenue']
    test_label  = test['revenue']


#    ['id', 'budget', 'runtime', 'revenue', 'release_month', 'release_day', 'release_year', 'release_dayofweek',
#     'release_quarter', 'high_budget_adjacency', 'is_collection', 'has_homepage', 'original_title_letter_count',
#     'original_title_word_count', 'overview_word_count', 'num_genres', 'genres_popularity', 'num_keywords',
#     'keywords_popularity', 'num_prod_companies', 'prod_companies_popularity', 'num_producers', 'producers_popularity',
#     'num_crew', 'num_director', 'director_popularity', 'num_prod_countries', 'prod_countries_popularity', 'language',
#     'num_spoken_lang', 'num_cast', 'num_females', 'is_lead_female', 'cast_popularity', 'top5_popularity',
#     'inflation_budget', 'scaled_budget', 'budget_num_cast_ratio', 'budget_runtime_ratio']

    final_dropped = ['id', 'revenue']
    tentative_drop0 = []
    tentative_drop1 = ['num_cast','cast_popularity','budget_num_cast_ratio']
    tentative_drop2 = ['num_director','director_popularity','num_crew']
    tentative_drop3 = ['release_month', 'release_day', 'release_dayofweek','release_quarter', 'is_collection', 'has_homepage', 'original_title_letter_count','original_title_word_count', 'overview_word_count', 'num_genres', 'num_keywords','keywords_popularity', 'num_prod_companies', 'prod_companies_popularity', 'num_producers', 'producers_popularity','num_crew', 'num_director', 'director_popularity', 'num_prod_countries', 'prod_countries_popularity', 'language','num_spoken_lang', 'num_cast', 'num_females', 'is_lead_female', 'cast_popularity', 'top5_popularity']
    tentative_drop4 = ['budget','high_budget_adjacency','inflation_budget', 'scaled_budget', 'budget_num_cast_ratio', 'budget_runtime_ratio']
    dropped = final_dropped + tentative_drop1
    train_data = train.drop(columns=dropped)
    test_data  = test.drop(columns=dropped)
    result = regression(train_data, test_data, train_label, test_label)

    for i,score in enumerate(result):
        if score < 0:
            result[i] = 0

    print("test mean: ", test_label.mean()*max_revenue)
    print("res  mean: ", result.mean() * max_revenue)
    final_baseline = test_label*max_revenue
    final_prediction = result*max_revenue
    rms = sqrt(mean_squared_error(final_baseline,final_prediction))
    print("RMSE: ", rms/100000000)
    print("Average mistake (%): ", 100*abs(final_prediction.mean() - final_baseline.mean())/final_baseline.mean())



if __name__ == "__main__":
    main()

