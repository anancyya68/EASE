# @File : RFE_EASE_AS_Evaluator.py
# @Time : 2024/6/4 10:46
# @Author :

import copy
import math
import os
import time
from model.utils import *
from keras.models import load_model
from model.networks import PreAttention,combined_model,NeuralNetwork
import numpy as np
import pandas as pd
import tensorflow as tf
# from distribution_shift import DistributionShiftEvaluator
# from feature_selection import FeatureSelector
import random
from model.incremental_learning import EWC
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer, Dense, Input,Reshape,Flatten,Concatenate,LSTM, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.ensemble import RandomForestRegressor


def write_config(config_class, file_path):
    """
    :param config_class: the Class of defined parameter
    :param file_path: save path
    :return:
    """
    with open(os.path.join(file_path,'parameter.txt'), 'w') as f:
        for attr in dir(config_class):
            if not attr.startswith("__") and not callable(getattr(config_class, attr)):
                value = getattr(config_class, attr)
                f.write(f"{attr} = {value}\n")
    f.close()

def write_parameter(file_path,fea_epoch,pre_epoch,num_heads,embed_dim):
    """
    :param config_class: the Class of defined parameter
    :param file_path: save path
    :return:
    """
    config_file = open(os.path.join(file_path, 'parameter.txt'), 'w')
    config_file.write('pre_epoch:' + str(pre_epoch) + '\n')
    config_file.write('fea_epoch:' + str(fea_epoch) + '\n')
    config_file.write('num_heads:' + str(num_heads) + '\n')
    config_file.write('embed_dim:' + str(embed_dim) + '\n')
    config_file.close()

def pre_training(data, Folder, pre_epoch, embed_dim, num_head,pre_lr):
    print('pre training start:')
    # 1.model
    output_shape = (1, embed_dim)
    model = combined_model(embed_dim, output_shape, num_head)
    print(model.summary())
    dummy_input = np.zeros((1, data.shape[1]-1, embed_dim))
    model.predict(dummy_input)
    adam = tf.keras.optimizers.Adam(learning_rate=pre_lr)
    model.compile(loss='mse', optimizer=adam)
    #print(model.summary())

    # 2. sample
    X_trains, y_trains = pre_trans_shape(data, embed_dim)
    train_idxs, val_idxs = pre_train_val_split(X_trains.shape[0])
    # Folder = create_res(Folder_path=Config.pre_save_path)
    i = 0
    for idx in train_idxs:
        print('{}_th data:'.format(i))
        X_train = X_trains[idx]
        y_train = y_trains[idx]
        val_idx = random.choice(val_idxs)
        X_val = X_trains[val_idx]
        y_val = y_trains[val_idx]

        # 3.weight decay strategy
        lr_scheduler = LearningRateScheduler(decay_schedule)

        # 4.train and save model
        checkpoint1 = ModelCheckpoint(os.path.join(Folder, 'best_0.tf'),
                                      monitor='val_loss', save_best_only=True,
                                      save_weights_only=False, mode='min', save_freq='epoch', verbose=2)
        print(X_train.shape)
        print(y_train.shape)
        history = model.fit(X_train, y_train, epochs=pre_epoch, batch_size=1, callbacks=[checkpoint1, lr_scheduler], validation_data=(X_val, y_val))

        #plot_loss(Folder, history,idx)
        i += 1
    # # 5.Evaluation
    # for val_id in val_idxs:
    #     X_val = X_trains[val_id]
    #     y_val = y_trains[val_id]
    #     model = load_model(os.path.join(Folder, 'best_0.tf'))
    #     y_pred = model.predict(X_val)
    #     mse,mae,rmae,r2 = regression_metrics(y_val,y_pred)
    #     write_regression(Folder, mse, mae, rmae, r2,val_id)
    #     plot_regression(Folder, y_val, y_pred,val_id)

def get_layer_by_partial_name(model, partial_name):
    matched_layers = []
    for layer in model.layers:
        if partial_name in layer.name:
            matched_layers.append(layer.name)
    if not matched_layers:
        raise ValueError(f"No layer containing '{partial_name}' found in the model.")
    return matched_layers

def get_feature_importance(model,current_num,embed_dim):
    matched_layer = get_layer_by_partial_name(model, 'tf.nn.softmax')
    attention_output_model = Model(inputs=model.input, outputs=model.get_layer(matched_layer[0]).output)
    sample_input = np.random.rand(1,current_num,embed_dim)
    attention_weights = attention_output_model.predict(sample_input)
    attention_weights_mean_heads = np.mean(attention_weights, axis=1)
   # feature_importance_mean = np.mean(attention_weights_mean_heads, axis=1)
    feature_importance_sum = np.sum(attention_weights_mean_heads, axis=1)
    return feature_importance_sum

def drop_lowwest_feature(data,feature_importance):
    """
    :param data:
    :param feature_importance:
    :return:
    """
    min_index = np.argmin(feature_importance)
    df = data.drop(data.columns[min_index], axis=1)
    return df

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
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y.values.reshape(-1, 1))
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

def get_EASE_data(data,Folder, embed_dim,fea_epoch,select_ratio,fea_lr=0.001):
    select_num = math.ceil((data.shape[1] - 1) * select_ratio)
    current_num = data.shape[1] - 1
    t = 0
    time_of_epoch = []
    total_losses = pd.DataFrame(columns=list(range(current_num)))
    while current_num > select_num:
        # load model
        start_time = time.time()
        model = load_model(os.path.join(Folder, 'best_{}.tf'.format(t)),
                           custom_objects={'PreAttention': PreAttention,
                                           'NeuralNetwork': NeuralNetwork})
        # delete unimportant features
        feature_importance = get_feature_importance(model, current_num, embed_dim)
        data = drop_lowwest_feature(data, feature_importance)
        current_num = data.shape[1] - 1
        print('current_num:', current_num)
        # incremental training
        X_train, y_train = trans_shape(data, embed_dim)
        ewc = EWC(model=model, lambda_ewc=1)
        ewc.compute_fisher_matrix(X_train, y_train)
        loss_list = ewc.train(X_train, y_train, Folder, t, epochs=fea_epoch, batch_size=embed_dim*2,
                              learning_rate=fea_lr)
        total_losses.iloc[:, t] = loss_list
        end_time = time.time()
        time_of_epoch.append(end_time - start_time)
        t = t + 1
    total_losses.to_csv(os.path.join(Folder, 'incremental_loss.csv'))
    pd.DataFrame(time_of_epoch).to_csv(os.path.join(Folder, 'time_of_epoch.csv'))
    return data


def get_regression_result(data_path, Folder_path, dataset, select_ratio, embed_dim, fea_epoch, num_head, pre_epoch,pre_lr=0.001,fea_lr=0.001):
    Folder = create_res(Folder_path, dataset)
    res = pd.DataFrame(index=list(range(times)), columns=['mae', 'mse', 'rmse', 'r2'])
    print('=================dataset:{}=================='.format(dataset))
    data = pd.DataFrame(pd.read_hdf(os.path.join(data_path, dataset)))
    data = standardize_regression(data)
    pre_training(data, Folder, pre_epoch, embed_dim, num_head,pre_lr)
    write_parameter(Folder, fea_epoch, pre_epoch, num_head, embed_dim)
    data = get_EASE_data(data, Folder, embed_dim, fea_epoch, select_ratio, fea_lr)
    data.to_csv(os.path.join(Folder, dataset + '.csv'))
    print('Predictor Task Start:')
    print(data.shape)
    candidate_rf = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
                    'max_depth': [2, 3, 4, 5, 6]}
    print('data shape:', data.shape)
    print(data.columns)
    df = copy.deepcopy(data)
    for i in range(times):
        X_train, X_val, X_test, y_train, y_test, y_val = split_data(df)
        best_predictor = parameter_search(X_val, y_val.ravel(), RandomForestRegressor(),
                                          candidate_rf)
        best_predictor.fit(X_train, y_train)
        # prediction
        y_pred = best_predictor.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        res.iloc[i, :] = [mae, mse, rmse, r2]
    res.loc['mean'] = res.mean()
    res.loc['std'] = res.std()
    res.to_csv(os.path.join(Folder, dataset + '_res.csv'))


if __name__ == "__main__":
    print('Regression Task:')
    data_path = './data/re'
    re_datasets = ['openml_586.hdf','openml_618.hdf','airfoil.hdf']
    Folder_path = './result/ex1/RFE_EASE'
    t = 0
    times = 5
    select_ratio = 0.7
    fea_epochs = [120,140,180]
    num_heads = [16, 32]
    pre_epochs = [10, 30, 50]
    embed_dims = [32,64]
    for dataset in re_datasets:
        for pre_epoch in pre_epochs:
            for fea_epoch in fea_epochs:
                for num_head in num_heads:
                    for embed_dim in embed_dims:
                        get_regression_result(data_path, Folder_path, dataset, select_ratio, embed_dim, fea_epoch, num_head,
                                          pre_epoch)











