# @File : 1.RFE_EASE.py
# @Time : 2024/6/11 17:22
# @Author :

import os
import math
import stat
import random
import datetime
import pandas as pd
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import load_model
from tensorflow.keras.models import Model
from model.incremental_learning import EWC
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from model.networks import combined_model,NeuralNetwork,PreAttention
from model.utils import *




def pre_train_val_split(n, ratio=0.7):
    """
    split data
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

def  pre_training(data, Folder, pre_epoch, pre_lr, embed_dim, num_head):
    print('pre training start:')
    num_classes = len(data.iloc[:,-1].unique())
    X_trains, y_trains = pre_trans_shape(data, embed_dim)
    train_idxs, val_idxs = pre_train_val_split(X_trains.shape[0])

    model = combined_model(embed_dim, num_classes, num_head)
    dummy_input = np.zeros((X_trains[0].shape[0], X_trains[0].shape[1], embed_dim))
    model.predict(dummy_input)
    print('************************')
    print(X_trains.shape)
    print(y_trains.shape)

    i = 0
    for idx in train_idxs:
        print('{}_th data:'.format(i))
        X_train = X_trains[idx]
        print(X_train.shape)
        y_train1 = y_trains[idx]
        print(y_train1.shape)
        val_idx = random.choice(val_idxs)
        X_val = X_trains[val_idx]
        y_val1 = y_trains[val_idx]
        y_train = to_categorical(y_train1, num_classes=num_classes)
        y_val = to_categorical(y_val1, num_classes=num_classes)

        lr_scheduler = LearningRateScheduler(decay_schedule)
        optimizer = tf.keras.optimizers.Adam(learning_rate=pre_lr)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        checkpoint1 = ModelCheckpoint(os.path.join(Folder, 'best_0.tf'), monitor='val_loss', save_best_only=True,
                                      save_weights_only=False, mode='min', save_freq='epoch', verbose=0)

        history = model.fit(X_train, y_train, epochs=pre_epoch, batch_size=1, callbacks=[checkpoint1, lr_scheduler],
                            validation_data=(X_val, y_val))
        i += 1

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

def trans_shape(data,row_num):
    df = data.sample(n=row_num, random_state=np.random.randint(10000))
    x_train = df.iloc[:, :-1]
    x_train1 = tf.transpose(x_train)
    X_train = tf.expand_dims(x_train1, axis=0)
    y_train = df.iloc[:,-1]
    y_train = np.array(y_train).reshape(1,row_num)
    return X_train, y_train

def get_EASE_data(data,Folder, pre_epoch, embed_dim, num_head,fea_epoch,select_ratio,fea_lr=0.001,pre_lr=0.001):
    num_classes = len(data.iloc[:, -1].unique())

    pre_training(data, Folder, pre_epoch, pre_lr, embed_dim, num_head)

    select_num = math.ceil((data.shape[1] - 1) * select_ratio)
    current_num = data.shape[1] - 1
    t = 0
    time_of_epoch = []
    total_losses = pd.DataFrame(columns=list(range(current_num)))
    while current_num > select_num:
        #  load previous model
        start_time = time.time()
        model = load_model(os.path.join(Folder, 'best_{}.tf'.format(t)),
                           custom_objects={'PreAttention': PreAttention,
                                           'NeuralNetwork': NeuralNetwork})
        #print(model.summary())
        # delete unimportant features
        feature_importance = get_feature_importance(model, current_num, embed_dim)


        data = drop_lowwest_feature(data, feature_importance)
        current_num = data.shape[1] - 1
        print('current_num:', current_num)

        # incremental training
        X_train, y_train = trans_shape(data, embed_dim)
        y_train = to_categorical(y_train, num_classes=num_classes)


        ewc = EWC(model=model, lambda_ewc=1)
        ewc.compute_fisher_matrix(X_train, y_train)
        loss_list = ewc.train(X_train, y_train, Folder, t, epochs=fea_epoch, batch_size=embed_dim,
                              learning_rate=fea_lr)
        total_losses.iloc[:,t] = loss_list

        end_time = time.time()
        time_of_epoch.append(end_time-start_time)

        # delete_file_name = os.path.join(Folder, 'best_{}.tf'.format(t))
        # if os.path.exists(delete_file_name):
        #     os.chmod(delete_file_name, stat.S_IWUSR | stat.S_IXUSR)
        #     os.remove(delete_file_name)
        t = t + 1
    total_losses.to_csv(os.path.join(Folder,'incremental_loss.csv'))
    pd.DataFrame(time_of_epoch).to_csv(os.path.join(Folder,'time_of_epoch.csv'))
    return data

def get_result(data_path,Folder_path,dataset,select_ratio,embed_dim,fea_epoch,num_head,pre_epoch):
    Folder = create_res(Folder_path, dataset)
    write_parameter(Folder, fea_epoch, pre_epoch, num_head, embed_dim, select_ratio)
    res = pd.DataFrame(index=list(range(5)), columns=['acc', 'pre', 'recall', 'f1'])
    print('=================dataset:{}=================='.format(dataset))
    data = pd.DataFrame(pd.read_hdf(os.path.join(data_path, dataset)))
    data = standardize_calssification(data)
    data = check_and_map_labels(data)
    data = get_EASE_data(data, Folder, pre_epoch, embed_dim, num_head, fea_epoch, select_ratio)
    print('Predictor Task Start:')
    data.to_csv(os.path.join(Folder, dataset + '.csv'))
    print(data.shape)
    candidate_rf = {
        'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
        'max_depth': [2, 3, 4, 5, 6]}
    print('data shape:', data.shape)
    print(data.columns)
    df = copy.deepcopy(data)
    for i in range(5):
        X_train, X_val, X_test, y_train, y_test, y_val = split_data(df)
        best_predictor = parameter_search(X_val, y_val.ravel(), RandomForestClassifier(), candidate_rf)
        best_predictor.fit(X_train, y_train)
        y_pred = best_predictor.predict(X_test)
        acc, pre, f1, recall = classification_metrics(y_test, y_pred)
        res.iloc[i, :] = [acc, pre, f1, recall]
    res.loc['mean'] = res.mean()
    res.loc['std'] = res.std()
    res.to_csv(os.path.join(Folder, dataset + '_res.csv'))


if __name__=='__main__':

    select_ratio = 0.7
    embed_dims = [32,64]
    fea_epochs = [100,110,120]
    num_heads = [4,8,16]
    pre_epochs = [30,40,50]
    print('Classification Task:')
    data_path = './data/class'
    Folder_path = './result/ex1/RFE_AFQE'
    datatsets = ['wine_red.hdf', 'svmguide3.hdf', 'spectf.hdf']
    for dataset in datatsets:
        print('dataset:',dataset)
        for embed_dim in embed_dims:
            for fea_epoch in fea_epochs:
                for num_head in num_heads:
                    for pre_epoch in pre_epochs:
                        get_result(data_path, Folder_path, dataset, select_ratio, embed_dim, fea_epoch, num_head, pre_epoch)









