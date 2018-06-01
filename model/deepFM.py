# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class deepFM(BaseEstimator, TransformerMixin):
    def __init__(self, args):
    """
        feature_size,
        field_size,
        embedding_dim,
        dropout_fm=[1.0, 1.0],
        deep_layers=[32, 32],
        dropout_deep=[0.5, 0.5, 0.5],
        epoch=10,
        batch_size=256,
        learning_rate=0.1,
        optimizer='Adam',
        batch_norm=0,
        batch_norm_decay=0.995,
        verbose=False,
        random_seed=2016,
        loss_type='mse',
        l2_regularization=0.0)
    """

        self.args = args

        # check parameters
        assert self.args.loss_type in ["logloss", "mse"], "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.graph = tf.Graph()
        with self.graph.as_defalut():
            self.feature_index = tf.placeholder(tf.int32, shape=[None, None], name='feature_index') # batch * feature_size
            self.feature_value = tf.placeholder(tf.float32, shape=[None, None], name='feature_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.droupout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='droupout_keep_fm')
            self.droupout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='droupout_keep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            self.weights = self._initialize_weights()

            # FM-model
            # ---------- first order term -------------
            self.y_first_order = tf.nn.embedding_lookup(self.weights['FM_W'], self.feature_index) # None * feature_size * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feature_value), 1) # None * feature_size
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.droupout_keep_fm[0])

            # ---------- second order term -------------
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feature_index) # None * feature_size * embedding_size
            #>>> feature_value != field_size >>>> ##############
            feature_value = tf.reshape(self.feature_value, shape=[-1, self.args.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feature_value)

            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)
            self.summed_features_emb_square = tf.square(self.summed_features_emb)

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.droupout_keep_fm[1])


            # ---------- deep component -------------
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.args.field_size * self.args.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep, droupout_keep_deep[0])
            for i in range(len(self.args.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights['layer_%d' % i]), self.weights['bias_%d' % i])
                if self.args.batch_norm:
                    pass
#                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn='bn_%d' % i)
                self.y_deep = tf.nn.relu(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.droupout_keep_deep[1 + i])

            if self.args.which_model == 'DeepFM':
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            self.out = tf.add(tf.matmul(concat_input, self.weights['concat_projection']), self.weights['concat_bias'])

            # loss
            if self.args.loss_type = 'mse':
                self.loss = tf.nn.l2_loss(tf.subract(self.label, self.out))
            # l2 regularization on weights
            if self.args.l2_regularization > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_regularization)(self.weights['concat_projection'])
                for i in range(len(self.args.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_regularization)(self.weights['layer_%d' % i])

            # optimizer
            if self.args.optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(self.args.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)


            # init
            self.saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

    def _initialize_weights(self):
        weights = dict()

        # embedding
        weights['FM_B'] = tf.get_variable(name='FM_B', shape=[1], initializer=tf.constant_initializer(0.0))
        weights['FM_W'] = tf.get_variable(name='FM_W', shape=[self.args.feature], initializer=tf.glorot_normal_initializer())
        weights['feature_embedding'] = tf.get_variable(name='feature_embedding', shape=[self.args.feature_size, self.args.embedding_size], initializer=tf.glorot_normal_initializer())

        # deep layers
        num_layers = len(self.args.deep_layers)
        input_size = self.args.field_size * self.args.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.args.deep_layers[0])
        weights['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.args.deep_layers[0])), dtype=np.float32)
        weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.args.deep_layers[0])), dtype=np.float32)

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.args.deep_layers[i-1] + self.args.deep_layers[i])
            weights['layer_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.args.deep_layers[i-1], self.args.deep_layers[i])), dtype=np.float32)
            weights['bias_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.args.deep_layers[i])), dtype=np.float32)

        if self.args.which_model == 'DeepFM':
            input_size = self.args.field_size + self.args.embedding_size + self.args.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=float32)

        return weights
