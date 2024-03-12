import pandas as pd
import numpy as np
from preprocess import preprocess_data, split_data_train, create_pools
from predict import train_model, prediction
from sklearn.metrics import ndcg_score


if __name__ == '__main__':
    train = pd.read_csv('./data/vk_train_df.csv')
    test = pd.read_csv('./data/vk_test_df.csv')
    train = preprocess_data(train)
    test = preprocess_data(test, is_train=False)
    X_train, X_eval, y_train, y_eval = split_data_train(train)
    train_pool = create_pools(X_train, y_train)
    eval_pool = create_pools(X_eval, y_eval)
    train_model((train_pool, eval_pool))
    predictions = prediction(test.drop(columns=['target']))
    print(ndcg_score(
        np.array([test['target'].values]),
        np.array([predictions])
        ))
