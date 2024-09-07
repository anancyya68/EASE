# @File : incremental_learning.py
# @Time : 2024/5/22 16:22
# @Author :

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

class EWC:
    def __init__(self, model, lambda_ewc=1):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_matrix = None
        self.initial_params = None

    def _get_grads(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = tf.keras.losses.mean_squared_error(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        return grads

    def compute_fisher_matrix(self, X_train, y_train):
        fisher_matrix = None
        for x, y in zip(X_train, y_train):
            grads = self._get_grads(np.expand_dims(x, axis=0), np.expand_dims(y, axis=0))
            if fisher_matrix is None:
                fisher_matrix = [np.zeros_like(g) if g is not None else None for g in grads]
            for i, g in enumerate(grads):
                if g is not None:
                    fisher_matrix[i] += g ** 2

        self.fisher_matrix = [f / len(X_train) if f is not None else None for f in fisher_matrix]
        self.initial_params = [tf.identity(w) for w in self.model.trainable_variables]

    def ewc_loss(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            base_loss = tf.keras.losses.mean_squared_error(y, logits)
            ewc_penalty = 0
            for i, param in enumerate(self.model.trainable_variables):
                if self.fisher_matrix[i] is not None:
                    ewc_penalty += tf.reduce_sum(self.fisher_matrix[i] * (param - self.initial_params[i]) ** 2)
            total_loss = base_loss + (self.lambda_ewc / 2) * ewc_penalty
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        return total_loss, grads

    def custom_loss(self, y_true, y_pred):
        loss, _ = self.ewc_loss(y_true, y_pred)
        return loss

    def train(self, new_X_train, new_y_train, Folder,t,epochs=10, batch_size=20, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        dataset = tf.data.Dataset.from_tensor_slices((new_X_train, new_y_train))
        dataset = dataset.batch(batch_size)
        loss_list = []
        for epoch in range(epochs):
            for step, (x_batch, y_batch) in enumerate(dataset):
                total_loss, grads = self.ewc_loss(x_batch, y_batch)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                if epoch % 30==0:
                    print(f'Epoch {epoch + 1}, Loss: {total_loss.numpy()}')
                loss_list.append(total_loss.numpy())
        checkpoint_path = os.path.join(Folder, 'best_{}.tf'.format(t + 1))
        tf.keras.models.save_model(self.model, checkpoint_path)
        return loss_list

    def train2(self, new_X_train, new_y_train, Folder,t,epochs=10, learning_rate=0.001):
        print(new_X_train.shape)
        print(new_y_train.shape)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=self.custom_loss)
        checkpoint1 = ModelCheckpoint(os.path.join(Folder, 'best_{}.tf'.format(t+1)), monitor='val_loss', save_best_only=True,
                                      save_weights_only=False, mode='min', period=1, verbose=0)
        self.model.fit(new_X_train, new_y_train, epochs=epochs,callbacks=[checkpoint1])
        return self.model





class Routine:
    def __init__(self, model):
        self.model = model

    def custom_loss(self, y_true, y_pred):
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:  # Regression problem
            return tf.keras.losses.mean_squared_error(y_true, y_pred)
        else:  # Classification problem
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    def train(self, new_X_train, new_y_train, Folder, t, epochs=10, batch_size=10, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        dataset = tf.data.Dataset.from_tensor_slices((new_X_train, new_y_train))
        dataset = dataset.batch(batch_size)
        loss_list = []
        loss_list1 = []
        no_change_epochs = 0
        prev_loss = None

        for epoch in range(epochs):
            epoch_loss = 0
            for step, (x_batch, y_batch) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch, training=True)
                    loss_value = self.custom_loss(y_batch, logits)
                    total_loss = tf.reduce_mean(loss_value)
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                epoch_loss += total_loss.numpy()
                loss_list1.append(total_loss.numpy())

            epoch_loss /= len(dataset)
            loss_list.append(epoch_loss)

            if prev_loss is not None and abs(prev_loss - epoch_loss) < 1e-6:
                no_change_epochs += 1
            else:
                no_change_epochs = 0
            prev_loss = epoch_loss

            if no_change_epochs >= 10:
                print(f"Training stopped early at epoch {epoch + 1} due to no change in loss.")
                break
        checkpoint_path = os.path.join(Folder, 'best_{}.tf'.format(t + 1))
        tf.keras.models.save_model(self.model, checkpoint_path)
        return loss_list1
