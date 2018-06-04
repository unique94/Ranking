# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class deepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, embedding_dim, field_multi_size=1, field_multi_padding_size=40, field_single_size=32,
                 dropout_fm=[1.0, 1.0], deep_layers=[64, 64], dropout_deep=[0.5, 0.5, 0.5],
                 epoch=10, batch_size=1, learning_rate=0.1, optimizer='Adam', batch_norm=0,
                 batch_norm_decay=0.995, verbose=False, random_seed=2018, loss_type='mse', l2_regularization=0.0):

        # check parameters
        assert loss_type in ["logloss", "mse"], "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size
        self.field_single_size = field_single_size
        self.field_multi_size = field_multi_size
        self.field_multi_padding_size = field_multi_padding_size
        self.field_size = field_multi_size + field_single_size
        self.embedding_dim = embedding_dim

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.l2_regularization = l2_regularization

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.feature_field_multi_index_ = tf.placeholder(tf.int32, shape=[None, None, None], name='feature_field_multi_index')  # batch, field_multi_size, feature_id + padding
            self.feature_field_multi_value_ = tf.placeholder(tf.int32, shape=[None, None, None], name='feature_field_multi_value')  # batch, field_multi_size, feature_id + padding
            self.feature_field_single_ = tf.placeholder(tf.int32, shape=[None, None], name='feature_field_single')  # batch, field_single_size
            self.label_ = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_fm_ = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
            self.dropout_keep_deep_ = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')

            # ????? what's this ? --->
            self.train_phase_ = tf.placeholder(tf.bool, name='train_phase')

            self.weights = self._initialize_weights()

            # ----FM-model----
            # ---------- first order term -------------
            self.multi_first_order = tf.nn.embedding_lookup(self.weights['FM_W'], self.feature_field_multi_index_)  # batch, field_multi_size, field_multi_padding_size
            self.multi_first_order = tf.reduce_sum(tf.multiply(self.multi_first_order, self.feature_field_multi_value_), 2, keepdims=True)  # batch, field_multi_size, 1
            feature_field_multi_value_sum = tf.reduce_sum(self.feature_field_multi_value_, 2, keepdims=True)
            self.multi_first_order = tf.reduce_sum(tf.div(self.multi_first_order, feature_field_multi_value_sum), 2)  # batch, field_multi_size

            self.single_first_order = tf.nn.embedding_lookup(self.weights['FM_W'], self.feature_field_single_)  # batch, field_single_size

            self.fm_first_order = tf.reduce_sum(tf.concat([self.multi_first_order, self.single_first_order], 1), 1)  # batch

            # ---------- second order term -------------
            self.multi_embeddings = tf.nn.embedding_lookup(self.weights['feature_embedding'], self.feature_field_multi_index_)  # batch, field_multi_size, field_multi_padding_size, embedding_dim
            feature_field_multi_value = tf.reshape(self.feature_field_multi_value_, shape=[-1, self.field_multi_size, self.field_multi_padding_size, 1])
            self.multi_embeddings = tf.multiply(self.multi_embeddings, feature_field_multi_value)  # batch, feature_multi_size, field_multi_padding_size, embedding_dim
            self.multi_embeddings = tf.div(tf.reduce_sum(self.multi_embeddings, 2), feature_field_multi_value_sum)  # batch, feature_multi_size, embedding_dim

            self.single_embeddings = tf.nn.embedding_lookup(self.weights['feature_embedding'], self.feature_field_single_)  # batch, field_single_size, embedding_dim
            self.embeddings = tf.concat([self.multi_embeddings, self.single_embeddings], 1)  # batch, field_size, embedding_dim

            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # batch, embedding_dim
            self.summed_features_emb_square = tf.square(self.summed_features_emb)

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # batch, embedding_dim

            self.fm_second_order = 0.5 * tf.reduce_sum(tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb), 1)  # batch

            # ----deep-model----
            self.deep_input = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_dim])
            self.deep_input = tf.nn.dropout(self.deep_input, self.dropout_keep_deep_[0])
            for i in range(len(self.deep_layers)):
                self.deep_input = tf.add(tf.matmul(self.deep_input, self.weights['layer_%d' % i]), self.weights['bias_%d' % i])
                if self.batch_norm:
                    pass
#                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn='bn_%d' % i)
                self.deep_input = tf.nn.relu(self.deep_input)
                self.deep_input = tf.nn.dropout(self.deep_input, self.dropout_keep_deep_[1 + i])

            self.deep_output = tf.add(tf.matmul(self.deep_input, self.weights['deep_final_w']), self.weights['deep_final_b'])

            # concat fm and deep
            self.output = tf.sigmoid(self.fm_first_order + self.fm_second_order + self.deep_output)

            # loss
            if self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label_, self.output))
            # l2 regularization on weights
            #!!! embeddings weights
            if self.l2_regularization > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_regularization)(self.weights['deep_final_w'])
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_regularization)(self.weights['layer_%d' % i])

            # optimizer
            if self.optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def _initialize_weights(self):
        weights = dict()

        # FM-weights
        weights['FM_W'] = tf.get_variable(name='FM_W', shape=[self.feature_size], initializer=tf.glorot_normal_initializer())
        weights['feature_embedding'] = tf.get_variable(name='feature_embedding', shape=[self.feature_size, self.embedding_dim], initializer=tf.glorot_normal_initializer())

        # deep layers
        num_layers = len(self.deep_layers)
        input_size = self.exist_feature_size * self.embedding_dim
        weights['layer_0'] = tf.get_variable(name='layer_0', shape=[input_size, self.deep_layers[0]], initializer=tf.glorot_normal_initializer())
        weights['bias_0'] = tf.get_variable(name='bias_0', shape=[1, self.deep_layers[0]], initializer=tf.glorot_normal_initializer())

        for i in range(1, num_layers):
            weights['layer_%d' % i] = tf.get_variable(name='layer_%d' % i, shape=[self.deep_layers[i - 1], self.deep_layers[i]], initializer=tf.glorot_normal_initializer())
            weights['bias_%d' % i] = tf.get_variable(name='bias_%d' % i, shape=[1, self.deep_layers[i]], initializer=tf.glorot_normal_initializer())

        weights['deep_final_w'] = tf.get_variable(name='deep_final_w', shape=[self.deep_layers[-1], 1], initializer=tf.glorot_normal_initializer())
        weights['deep_final_b'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def fit_one_batch(self, X_multi_index, X_multi_value, X_single, y):
        feed_dict = {self.feature_field_multi_index_: X_multi_index,
                     self.feature_field_multi_value_: X_multi_value,
                     self.feature_field_single_: X_single,
                     self.label_: y,
                     self.dropout_keep_fm_: self.dropout_fm,
                     self.dropout_keep_deep_: self.dropout_deep,
                     self.train_phase_: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


if __name__ == '__main__':
    model = deepFM(feature_size=10, exist_feature_size=3, embedding_dim=2)
    X_multi_index = np.array([[[1, 3, 5]], [[3, 0, 0]]])
    X_multi_value = np.array([[[1, 1, 1]], [[1, 0, 0]]])
    X_single = np.array([[6, 9], [7, 8]])
    y = np.array([[1], [0]])
    print(model.fit_one_batch(X_multi_index, X_multi_value, X_single, y))
