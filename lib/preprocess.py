import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from config import Variables
from catboost import Pool


def preprocess_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df.drop(columns=Variables.UNARY_COLUMNS, inplace=True)
    df.drop(columns=Variables.INSIGNIFICANT_COLUMNS, inplace=True)
    df.drop(columns=Variables.HIGH_CORRELATION_COLUMNS, inplace=True)

    cols = df.columns
    inds = [col for col in range(1, len(cols) - 1) if cols[col] not in Variables.CATEGORIAL_COLUMNS]
    if is_train:
        df.drop_duplicates(inplace=True)
        scaler = StandardScaler()
        df = df.values
        df[:, inds] = scaler.fit_transform(df[:, inds])
        with open('scaler_config.pkl', 'wb') as file:
            pickle.dump(scaler, file)
    else:
        with open('scaler_config.pkl', 'rb') as file:
            scaler = pickle.load(file)
        df = df.values
        df[:, inds] = scaler.transform(df[:, inds])
    df = pd.DataFrame(df, columns=cols)
    for col in Variables.CATEGORIAL_COLUMNS + ['search_id', 'target']:
        df[col] = df[col].astype(int)
    return df


def split_data_train(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    cutoff_id = df['search_id'].quantile(0.9)
    X_train = df.loc[df.search_id < cutoff_id].drop(columns=['target'])
    X_eval = df.loc[df.search_id >= cutoff_id].drop(columns=['target'])
    y_train = df.loc[df.search_id < cutoff_id]['target']
    y_eval = df.loc[df.search_id >= cutoff_id]['target']
    return X_train, X_eval, y_train, y_eval


def create_pools(X: pd.DataFrame, y: pd.DataFrame) -> Pool:
    weights = [0.75 if i == 1 else 0.25 for i in y]
    pool = Pool(data=X,
                label=y,
                cat_features=Variables.CATEGORIAL_COLUMNS,
                group_id=X['search_id'],
                weight=weights
                )
    return pool
