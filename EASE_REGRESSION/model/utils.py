import os
import math
import random
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from model.distribution_shift import DistributionShiftEvaluator
from sklearn.preprocessing import StandardScaler


def trans_shape(data,row_num):
    df = data.sample(n=row_num, random_state=np.random.randint(10000))
    x_train = df.iloc[:, :-1]
    x_train1 = tf.transpose(x_train)
    X_train = tf.expand_dims(x_train1, axis=0)
    y_train = df.iloc[:,-1]
    y_train = np.array(y_train).reshape(1,row_num)
    return X_train, y_train


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

def pre_train_val_split(n, ratio=0.7):
    """
    :param n:
    :param ratio:
    :return:
    """
    numbers = list(range(n))
    random.shuffle(numbers)
    split_index = math.floor(n * ratio)
    train_idxs = numbers[:split_index]
    val_idxs = numbers[split_index:]
    return train_idxs, val_idxs

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

def standardize_regression(df):
    """
    Standardize features and labels
    """
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]
    features = df[feature_cols]
    labels = df[[label_col]]  # Ensuring labels are in DataFrame format
    scaler_features = StandardScaler()
    scaler_labels = StandardScaler()
    scaled_features = scaler_features.fit_transform(features)
    scaled_labels = scaler_labels.fit_transform(labels)
    scaled_features_df = pd.DataFrame(scaled_features, columns=feature_cols)
    scaled_labels_df = pd.DataFrame(scaled_labels, columns=[label_col])
    scaled_data = pd.concat([scaled_features_df, scaled_labels_df], axis=1)
    return scaled_data

def split_data(data_path,dataset):
    """
    :param data_path:
    :param dataset:
    :return:
    """
    df = pd.read_hdf(os.path.join(data_path, dataset))
    df = pd.DataFrame(df)
    df = standardize_regression(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_val, X_test, y_train, y_test, y_val = dataset_divide(X, y)
    return X_train, X_val, X_test, y_train, y_test, y_val

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

def regression_metrics(y_test,y_pred):
    """
    :param y_test: test label
    :param y_pred: predicted label
    :return: mse,mae,r2,rmae
    """
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 =  r2_score(np.array(y_test).flatten(), np.array(y_pred).flatten())
    return mse,mae,rmse,r2

def write_regression(Folder, mse,mae,rmae,r2,i):
    """
    :param Folder: file path
    :param mse: mean_squared_error
    :param mae: mean_absolute_error
    :param rmae: root mean_absolute_error
    :param r2: r2_score
    :return:
    """
    result_file = open(os.path.join(Folder, 'result_{}.txt'.format(i)), 'w')
    result_file.write('mean_squared_error:' + str(mse) + '\n')
    result_file.write('mean_absolute_error:' + str(mae) + '\n')
    result_file.write('root_mean_absolute_error:' + str(rmae) + '\n')
    result_file.write('r2_score:' + str(r2) + '\n')
    result_file.close()

def plot_regression(Folder,y_test,y_pred,i):
    """
    :param Folder: file path
    :param y_test: test label
    :param y_pred: predicted label
    :return:
    """
    plt.plot(np.array(y_test).flatten(), 'r', label='pred')
    plt.plot(np.array(y_pred).flatten(), 'b', label='real')
    plt.xlabel('Sample number')
    plt.legend(loc='best')
    plt.savefig(os.path.join(Folder, 'res_{}.png'.format(i)), format='png', dpi=200)
    plt.close()

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

