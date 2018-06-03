# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class deepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, exist_feature_size, embedding_dim,
                 dropout_fm=[1.0, 1.0], deep_layers=[64, 64], dropout_deep=[0.5, 0.5, 0.5],
                 epoch=10, batch_size=1, learning_rate=0.1, optimizer='SGD', batch_norm=0,
                 batch_norm_decay=0.995, verbose=False, random_seed=2018, loss_type='mse', l2_regularization=0.0):

        # check parameters
        assert loss_type in ["logloss", "mse"], "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size
        self.exist_feature_size = exist_feature_size
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
            self.feature_index_ = tf.placeholder(tf.int32, shape=[None, None], name='feature_index')  # batch * exist_feature_size
            self.feature_value_ = tf.placeholder(tf.float32, shape=[None, None], name='feature_value')  # batch * exist_feature_size
            self.label_ = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_fm_ = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
            self.dropout_keep_deep_ = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')
            # ????? what's this ? --->
            self.train_phase_ = tf.placeholder(tf.bool, name='train_phase')

            self.weights = self._initialize_weights()

            # FM-model
            # ---------- first order term -------------
            self.y_first_order = tf.nn.embedding_lookup(self.weights['FM_W'], self.feature_index_)  # batch_size * each_exist_feature_size * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, self.feature_value_), 2)  # batch_size * exist_feature_size
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm_[0])  # batch_size * exist_feature_size

            # ---------- second order term -------------
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embedding'], self.feature_index_)  # batch_size * exist_feature_size * embedding_dim
            feature_value = tf.reshape(self.feature_value_, shape=[-1, self.exist_feature_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feature_value)  # batch_size * feature_size * embedding_dim

            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # batch_size * embedding_dim
            self.summed_features_emb_square = tf.square(self.summed_features_emb)

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # batch_size * embedding_dim

            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # batch_size * embedding_dim
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm_[1])

            # deep-model
            self.deep_input = tf.reshape(self.embeddings, shape=[-1, self.exist_feature_size * self.embedding_dim])
            self.deep_input_dropout = tf.nn.dropout(self.deep_input, self.dropout_keep_deep_[0])
            layer1 = tf.layer.dense(
                inputs=self.deep_input_dropout,
                units=self.deep_layers[0],
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_normal_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_regularization),
                name='deep_layer1'
            )
            layer1_drop = tf.layers.dropout(
                layer1,
                rate=self.dropout_keep_deep_[1]
            )
            layer2 = tf.layer.dense(
                inputs=layer1_drop,
                units=self.deep_layers[1],
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_normal_initializer(),
                activity_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_regularization),
                name='deep_layer2'
            )
            layer2_drop = tf.layers.dropout(
                layer2,
                rate=self.dropout_keep_deep_[2]
            )

            full_connect_layer = tf.layer.dense(

            )

            # loss
            if self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label_, self.out)) + tf.losses.get_regularization_loss()

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
        weights['FM_W'] = tf.get_variable(name='FM_W', shape=[self.feature_size, 1], initializer=tf.glorot_normal_initializer())
        weights['feature_embedding'] = tf.get_variable(name='feature_embedding', shape=[self.feature_size, self.embedding_dim], initializer=tf.glorot_normal_initializer())

        # deep layers
        num_layers = len(self.deep_layers)
        input_size = self.exist_feature_size * self.embedding_dim
        weights['layer_0'] = tf.get_variable(name='layer_0', shape=[input_size, self.deep_layers[0]], initializer=tf.glorot_normal_initializer())
        weights['bias_0'] = tf.get_variable(name='bias_0', shape=[1, self.deep_layers[0]], initializer=tf.glorot_normal_initializer())

        for i in range(1, num_layers):
            weights['layer_%d' % i] = tf.get_variable(name='layer_%d' % i, shape=[self.deep_layers[i - 1], self.deep_layers[i]], initializer=tf.glorot_normal_initializer())
            weights['bias_%d' % i] = tf.get_variable(name='bias_%d' % i, shape=[1, self.deep_layers[i]], initializer=tf.glorot_normal_initializer())

        input_size = self.exist_feature_size + self.embedding_dim + self.deep_layers[-1]
        weights['concat_projection'] = tf.get_variable(name='concat_projection', shape=[input_size, 1], initializer=tf.glorot_normal_initializer())
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feature_index_: Xi,
                     self.feature_value_: Xv,
                     self.label_: y,
                     self.dropout_keep_fm_: self.dropout_fm,
                     self.dropout_keep_deep_: self.dropout_deep,
                     self.train_phase_: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


if __name__ == '__main__':
    model = deepFM(feature_size=10, exist_feature_size=3, embedding_dim=2)
    Xi = np.array([[1, 4, 7]])
    Xv = np.array([[1, 1, 1]])
    y = np.array([[1], [0], [1]])
    print(model.fit_on_batch(Xi, Xv, y))
