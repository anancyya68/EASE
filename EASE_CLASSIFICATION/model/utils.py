import os
import math
import random
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import *
from model.distribution_shift import DistributionShiftEvaluator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,KFold


def standardize_calssification(df):
    """
    standardize features
    """
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]
    features = df[feature_cols]
    labels = df[label_col]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=feature_cols)
    scaled_data = pd.concat([scaled_features_df, labels.reset_index(drop=True)], axis=1)
    return scaled_data

def check_and_map_labels(df):
    """
    Map labels from [1, 2, ..., num_class] to [0, 1, ..., num_class - 1].
    """
    unique_labels = df.iloc[:, -1].unique()
    num_classes = len(unique_labels)
    if not df.iloc[:, -1].between(0, num_classes - 1).all():
        label_mapping = {label: index for index, label in enumerate(sorted(unique_labels))}
        df.iloc[:, -1] = df.iloc[:, -1].map(label_mapping)
    return df

def trans_shape(data,row_num):
    df = data.sample(n=row_num, random_state=np.random.randint(10000))
    x_train = df.iloc[:, :-1]
    x_train1 = tf.transpose(x_train)
    X_train = tf.expand_dims(x_train1, axis=0)
    y_train = df.iloc[:,-1]
    y_train = np.array(y_train).reshape(1,row_num)
   # y_train = y_train.reshape(-1)
    return X_train, y_train

def fea_trans_shape(data, row_num):
    """
    :param data: dataframe
    :param row_num: number of rows
    :return:
    """
    # shuffle the data
    data = data.sample(frac=1, random_state=np.random.randint(10000)).reset_index(drop=True)

    # Split data
    train_data = data.iloc[:row_num, :]
    val_data = data.iloc[row_num:2 * row_num, :] if len(data) >= 2 * row_num else data.iloc[row_num:, :].append(
        data.iloc[:2 * row_num - len(data), :])

    # training
    x_train = train_data.iloc[:, :-1]
    x_train1 = tf.transpose(x_train)
    X_train = tf.expand_dims(x_train1, axis=0)
    y_train = train_data.iloc[:, -1]
    y_train = np.array(y_train).reshape(1, row_num)

    # data
    x_val = val_data.iloc[:, :-1]
    x_val1 = tf.transpose(x_val)
    X_val = tf.expand_dims(x_val1, axis=0)
    y_val = val_data.iloc[:, -1]
    y_val = np.array(y_val).reshape(1, row_num)

    return X_train, y_train, X_val, y_val

def pre_trans_shape(data, row_num):
    """
    :param data:
    :param row_num:
    :return:
    """
    X_trains = []
    y_trains = []
    i = 0
    while i < len(data):
        start_idx = i
        end_idx = start_idx + row_num
        if end_idx > len(data):
            df_slice = data.iloc[-row_num:, :]
        else:
            df_slice = data.iloc[start_idx:end_idx, :]
        x_train = df_slice.iloc[:, :-1]
        x_train1 = tf.transpose(x_train)
        X_train = tf.expand_dims(x_train1, axis=0)
        y_train = df_slice.iloc[:, -1]
        y_train = np.array(y_train).reshape(1, row_num)
        X_trains.append(X_train.numpy())
        y_trains.append(y_train)
        i += row_num
    return np.array(X_trains), np.array(y_trains)

def create_res(Folder_path,dataset):
    """
    create save path
    :param Folder_path:
    :return:
    """
    current = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    Folder = os.path.join(Folder_path, os.path.join(dataset,current))

    if not os.path.exists(Folder):
        os.makedirs(Folder)
    return Folder

def dataset_divide(samples,targets,test_size=0.3,validation_size=0.5):
    """
    :param samples: samples
    :param targets: label
    :param test_size: test dataset ratio
    :param validation_size: validation dataset ratio
    :return: train dataset、validation dataset、test dataset
    """
    X_train, X_temp, y_train, y_temp = train_test_split(samples, targets, test_size=test_size, random_state=random.randint(1, 1000),shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=random.randint(1, 1000),shuffle=True)
    return X_train,X_val, X_test,y_train,y_test,y_val

def split_data(df):
    """
    :param data_path:
    :param dataset:
    :return:
    """
    df = pd.DataFrame(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_val, X_test, y_train, y_test, y_val = dataset_divide(X, y)
    return X_train, X_val, X_test, y_train, y_test, y_val

def parameter_search(X_val, y_val, predictor, candidate_params, n_splits=5):
    """
    :param X_val: features
    :param y_val: label
    :param train_algorithm: algorithm
    :param train_parameters: parameters
    :return select_parameters:parameters
    """
    best = GridSearchCV(predictor,candidate_params,cv=KFold(n_splits=n_splits, shuffle=True, random_state=22))
    best.fit(X_val, y_val)
    return best.best_estimator_

def write_parameter(file_path,fea_epoch,pre_epoch,num_head,embed_dim,select_ratio):
    """
    :param config_class: the Class of defined parameter
    :param file_path: save path
    :return:
    """
    config_file = open(os.path.join(file_path, 'parameter.txt'), 'w')
    config_file.write('pre_epoch:' + str(pre_epoch) + '\n')
    config_file.write('fea_epoch:' + str(fea_epoch) + '\n')
    config_file.write('num_heads:' + str(num_head) + '\n')
    config_file.write('embed_dim:' + str(embed_dim) + '\n')
    config_file.write('select_ratio:' + str(select_ratio) + '\n')
    config_file.close()

def write_config(config_class, file_path):
    """
    :param config_class: the Class of defined parameter
    :param file_path: save path
    :return:
    """
    with open(file_path, 'w') as f:
        for attr in dir(config_class):
            if not attr.startswith("__") and not callable(getattr(config_class, attr)):
                value = getattr(config_class, attr)
                f.write(f"{attr} = {value}\n")
    f.close()

def write_feature(file_path,t,columns):
    """
    :param file_path:
    :return:
    """
    with open(os.path.join(file_path,'feature.txt'), 'a') as f:
        f.write("The {}_th learning:".format(t) + '\n')
        f.write(str(columns))
        f.write('\n')
    f.close()

def decay_schedule(epoch, lr):
    """
    :param epoch: epoch
    :param lr: learning rate
    :return: decay lr
    """
    if (epoch % 300 == 0) and (epoch != 0):
        lr = lr * 0.95
    return lr

def plot_loss(Folder, history,i):
    """
    :param Folder: folder path
    :param history: csv,loss
    :return: figs
    """
    loss = pd.DataFrame(history.history['loss'])
    loss.to_csv(os.path.join(Folder, 'Loss.csv'))
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    fig_name = os.path.join(Folder, 'Loss_{}.png'.format(i))
    plt.savefig(fig_name, dpi=600)
    plt.close()

def classification_metrics(y_test,y_pred):
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred,average='macro')
    f1 = f1_score(y_test, y_pred,average='macro')
    recall = recall_score(y_test, y_pred,average='macro')
    return acc,pre,f1,recall

def write_classification(Folder, acc,pre,f1,recall,i):
    result_file = open(os.path.join(Folder, 'result_{}.txt'.format(i)), 'w')
    result_file.write('accuracy:' + str(acc) + '\n')
    result_file.write('precision:' + str(pre) + '\n')
    result_file.write('f1 score:' + str(f1) + '\n')
    result_file.write('recall:' + str(recall) + '\n')
    result_file.close()

def compare_data(data_previous,data_current,threshold):
    """
    :param data_previous: data t_1
    :param data_current: data t
    :param threshold:
    :return:
    """
    diff = DistributionShiftEvaluator(data_previous, data_current)
    val = diff.compute_js_divergence()
    print(val)
    return val > threshold

def fea_plot_loss(Folder,loss,t,fea_num):
    """
    :param loss:
    :return:
    """
    plt.plot(np.array(loss).flatten(), 'r', label='Feature_{}_num){}'.format(t,fea_num))
    plt.xlabel('Iteration number')
    plt.legend(loc='best')
    plt.savefig(os.path.join(Folder, 'fea_{}.png'.format(t)), format='png', dpi=200)
    plt.close()

