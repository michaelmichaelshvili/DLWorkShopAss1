import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from collections import Counter

def train_data_size():
    print('train data size: ' + str(len(os.listdir(r'C:\Users\micha\PycharmProjects\DLWorkShopAss1\dog-breed-identification\train'))))
    print('test data size: ' + str(len(os.listdir(r'C:\Users\micha\PycharmProjects\DLWorkShopAss1\dog-breed-identification\test'))))

def dimentions_histogram(images_path):
    dimensions = {}
    for idx, pic in enumerate(os.listdir(images_path)):
        print(idx)
        img = cv2.imread(os.path.join(images_path,pic))
        pic_dimensions_tuple = img.shape
        if(pic_dimensions_tuple[2] != 3):
            print(pic_dimensions_tuple)
            cv2.imshow(img)
        pic_dimensions = str(pic_dimensions_tuple)
        if pic_dimensions not in dimensions.keys():
            dimensions[pic_dimensions] = 0
        dimensions[pic_dimensions] += 1
    print(dimensions)
    k = Counter(dimensions)
    b = k.most_common()[:40]
    b = {i[0]: i[1] for i in b}
    f, ax = plt.subplots(figsize=(30, 15))
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    bars = plt.bar(b.keys(), b.values(), color='r')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval, rotation=90, fontsize=25)
    plt.xticks(rotation='vertical')
    plt.show()

def class_histogram():
    LABELS = r"/dog-breed-identification/labels.csv"

    train_df = pd.read_csv(LABELS)
    # return top 16 value counts and convert into list
    plt.figure(figsize=(20, 8))
    train_df['breed'].value_counts().plot(kind='bar')
    plt.show()

    top_breeds = sorted(list(train_df['breed'].value_counts().head(16).index))
    train_df = train_df[train_df['breed'].isin(top_breeds)]
    print(top_breeds)
    print('different between maximum and minimum: {}'.format(train_df['breed'].value_counts().head(1).sum() - train_df['breed'].value_counts().tail(1).sum()))

def show_breed(breed_name, num_of_imgs):
    LABELS = r"dog-breed-identification/labels.csv"
    train_df = pd.read_csv(LABELS)
    breed_list = train_df['breed'].unique()
    train_df = train_df.loc[train_df['breed'] == breed_name]
    for i in range(num_of_imgs):
        img = cv2.imread(os.path.join(r'dog-breed-identification\train',train_df.iloc[i]['id'] + r".jpg"))
        cv2.imshow(breed_name + str(i),img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# train_data_size()
# class_histogram()
# dimentions_histogram(r'C:\Users\micha\PycharmProjects\DLWorkShopAss1\dog-breed-identification\test')
show_breed('scottish_deerhound',5)
show_breed('japanese_spaniel',5)