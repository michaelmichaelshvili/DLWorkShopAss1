import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def train_data_size():
    print('train data size: ' + str(len(os.listdir(r'C:\Users\micha\PycharmProjects\DLWorkShopAss1\dog-breed-identification\train'))))
    print('test data size: ' + str(len(os.listdir(r'C:\Users\micha\PycharmProjects\DLWorkShopAss1\dog-breed-identification\test'))))

def dimentions_histogram():
    pass

def class_histogram():
    LABELS = r"../dog-breed-identification/labels.csv"

    train_df = pd.read_csv(LABELS)
    # return top 16 value counts and convert into list
    plt.figure(figsize=(20, 8))
    train_df['breed'].value_counts().plot(kind='bar')
    plt.show()

    top_breeds = sorted(list(train_df['breed'].value_counts().head(16).index))
    train_df = train_df[train_df['breed'].isin(top_breeds)]
    print(top_breeds)
    print('different between maximum and minimum: {}'.format(train_df['breed'].value_counts().head(1).sum() - train_df['breed'].value_counts().tail(1).sum()))
# train_data_size()
class_histogram()