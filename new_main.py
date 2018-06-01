# -*- coding: utf-8 -*-

import argparse

#config
parse = argparse.ArgumentParser(description='ranking')

parser.add_argument('--epochs', dest='epochs', type=int, default=10)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256)
parser.add_argument('--test_interval', dest='test_interval', type=int, default=100)

# random
parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=True)
parser.add_argument('--random_seed', dest='random_seed', type=int, default=66)

parser.add_argument('--field_size', dest='field_size', type=int)
parser.add_argument('--feature_size', dest='feature_size', type=int)
parser.add_argument('--embedding_size', dest='embedding_size', type=int, default=50)

parser.add_argument('--which_model', dest='which_model', type=str, default='DeepFM')
parser.add_argument('--loss_type', dest='loss_type', type=str, default='mse')
parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam')
parser.add_argument('--l2_regularization', dest='l2_regularization', type=float, default=1e-6)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1.0)



