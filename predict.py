import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from config import Variables


def train_model(data: tuple[Pool]) -> None:
    train_pool, eval_pool = data
    model = CatBoostRanker(**Variables.parameters)
    model.fit(train_pool, eval_set=eval_pool, verbose=1)
    model.save_model('/model.cbm')


def prediction(X_test: pd.DataFrame) -> np.array:
    model = CatBoostRanker()
    model.load_model('/model.cbm')
    predictions = model.predict(X_test)
    return predictions
