import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
import scipy.io


def to_data_frame(ds_X, ds_Y):
    df = pd.DataFrame(data = ds_X[:, 0])
    df[1] = ds_X[:, 1]
    df[2] = ds_Y[:, 0]
    df.columns = ["x1", "x2", "y"]
    return df


def count_error(predicted_values, real_values):
    wrong = 0
    for predicted_value, real_value in zip(predicted_values, real_values):
        if predicted_value != real_value:
            wrong += 1
    return wrong / len(predicted_values)
