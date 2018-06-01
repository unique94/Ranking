# -*- coding: utf-8 -*-

import argparse
import random
import torch
import dataset
import model
import train

parser = argparse.ArgumentParser(description='classifier')
# common
parser.add_argument('--epochs', dest='epochs', type=int, default=7)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1)
parser.add_argument('--log_interval', dest='log_interval', type=int, default=1)
parser.add_argument('--test_interval', dest='test_interval', type=int, default=100)

# random
parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=True)
parser.add_argument('--random_seed', dest='random_seed', type=int, default=66)
parser.add_argument('--torch_seed', dest='torch_seed', type=int, default=66)
# model parameters
parser.add_argument('--lr', dest='lr', type=float, default=0.1)
parser.add_argument('--lr_scheduler', dest='lr_scheduler', type=str, default='lambda')
parser.add_argument('--dropout_embed', dest='dropout_embed', type=float, default=0.5)
parser.add_argument('--dropout_rnn', dest='dropout_rnn', type=float, default=0.5)
parser.add_argument('--label_num', dest='label_num', type=int, default=3)
parser.add_argument('--p', dest='p', type=int, default=0)
parser.add_argument('--q', dest='q', type=int, default=1)
#
parser.add_argument('--use_embedding', dest='use_embedding', action='store_true', default=True)
parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=300)
parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=300)
# cuda
parser.add_argument('--cuda', dest='cuda', action='store_true', default=False)
#
parser.add_argument('--max_norm', dest='max_norm', type=float, default=None)
#
parser.add_argument('--which_dataset', dest='which_dataset', type=str, default='restaurant')
parser.add_argument('--which_model', dest='which_model', type=str, default='AT_TD_LSTM')
parser.add_argument('--load_model', dest='load_model', action='store_true', default=False)
parser.add_argument('--which_optim', dest='which_optim', type=str, default='SGD')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-6)

args = parser.parse_args()

torch.manual_seed(args.torch_seed)

data = dataset.MyDatasets(args)

args.embed_num = len(data.vocabulary.word2id)
args.aspect_num = len(data.vocabulary.word2id)
args.label_num = 3
args.cuda = args.cuda and torch.cuda.is_available()
print('\nParameters:')
for attr, value in sorted(args.__dict__.items()):
    print('\t{} = {}'.format(attr.upper(), value))

# model
m_model = None
if args.which_model == 'AT_TD_LSTM':
    m_model = model.AT_TD_LSTM(args, data)
    if args.load_model:
        m_model.load_state_dict(torch.load('./parameters/best_params/{}_best_params.pkl'.format(args.which_dataset)))
elif args.which_model == 'AT_TD_LSTM2':
    m_model = model.AT_TD_LSTM2(args, data)
elif args.which_model == 'AT_TD_LSTM3':
    m_model = model.AT_TD_LSTM3(args, data)
elif args.which_model == 'TD_LSTM':
    m_model = model.TD_LSTM(args, data)

if args.cuda:
    m_model.cuda()

print('\n', m_model)

## train and predict
torch.set_num_threads(1)
train.train(args, m_model, data.train_iterator, data.test_iterator)
