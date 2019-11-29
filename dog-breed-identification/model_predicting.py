from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import os


def load_model_from_path(path_to_model):
    model = load_model(path_to_model)
    return model


def predict_picture_class(classifier_model, path_to_img):
    img_data = cv2.imread(path_to_img)
    img_data = np.expand_dims(img_data, axis=0)
    prediction = classifier_model.predict(img_data, steps=1)
    return prediction


def get_breed_from_prediction(classes_list, prediction_vec):
    max_idx = np.argmax(prediction_vec)
    return str(classes_list[max_idx])


def predict_on_val_to_csv(path_to_image_dir, classfier_model, classes_list, label_path):
    label_df = pd.read_csv(label_path)
    predictions_df = pd.DataFrame(columns=['id','predicted','actual'])
    for idx, img in enumerate(os.listdir(path_to_image_dir)):
        print(idx)
        pred = predict_picture_class(classfier_model, os.path.join(path_to_image_dir, img))
        actual_lbl = str(list(label_df.loc[label_df['id'] == img[:-4]]['breed'])[0])
        predictions_df = predictions_df.append({'id' :img[:-4] , 'predicted' : get_breed_from_prediction(classes_list, pred[0]),'actual': actual_lbl}, ignore_index=True)
        if idx == 20:
            break
    predictions_df.to_csv('validation_score.csv')


def predict_to_csv(path_to_image_dir, classfier_model, classes_list):
    with open('sample_submission1.csv','w') as file:
        file.write('id,' + ','.join(classes_list)+'\n')
        for idx, img in enumerate(os.listdir(path_to_image_dir)):
            print(idx)
            pred = predict_picture_class(classfier_model, os.path.join(path_to_image_dir,img))
            file.write(img[:-4]+','+','.join(map(str,pred[0]))+'\n')

MODEL_PATH = r"..\models\weights_3_43.h5"
PICTURE_PATH = r"..\dog-breed-identification\test_resize\0a0b97441050bba8e733506de4655ea1.jpg"
LABELS_PATH = r"..\dog-breed-identification\labels.csv"
CLASSES_LIST = list(pd.read_csv(LABELS_PATH)['breed'].str.get_dummies())

# predict_to_csv(r"..\dog-breed-identification\test_resize",load_model_from_path(MODEL_PATH), CLASSES_LIST)
# pred = predict_picture_class(load_model_from_path(MODEL_PATH), PICTURE_PATH)
# print(get_breed_from_prediction(LABELS_PATH, pred))
predict_on_val_to_csv(r"../dog-breed-identification/train_resize", load_model_from_path(MODEL_PATH), CLASSES_LIST, LABELS_PATH)