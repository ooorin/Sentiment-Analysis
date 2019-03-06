# coding: utf-8
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import random
import gensim
import os

import models

import argparse

def str2bool(s):
	if s.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif s.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else: raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description='Four deep learning models.')
parser.add_argument('-model', type=str, default='FastText', \
	choices=['RNN', 'LSTM', 'FastText', 'CNN'], \
	help='Choose a model in [RNN, LSTM, FastText, CNN], default is FastText.')
parser.add_argument('-train', type=str2bool, default=False, \
	help='Whether to train the model, default is False.')
parser.add_argument('-predict', type=str2bool, default=True, \
	help='Whether to predict the sentence you input, default is True.')
parser.add_argument('-pre_train', type=str2bool, default=False, \
	help='Whether to use pre-trained model, default is True.')
args = parser.parse_args()

print(args)

SEED = 2333
torch.manual_seed(SEED)

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

if args.model == 'FastText':
	TEXT = data.Field(preprocessing=generate_bigrams)
else:
	TEXT = data.Field()
LABEL = data.LabelField(dtype=torch.float)

fields = {'label': ('label', LABEL), 'text': ('text', TEXT)}
train_data, test_data = data.TabularDataset.splits(
							path = 'data',
							train = 'train.json',
							test = 'test.json',
							format = 'json',
							fields = fields
	)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
	(train_data, valid_data, test_data),
	batch_size = BATCH_SIZE,
	)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256 # for RNN
N_FILTERS = 100 # for CNN
FILTER_SIZES = [3,4,5] # for CNN
OUTPUT_DIM = 1
N_LAYERS = 2 # for LSTM
BIDIRECTIONAL = True # for LSTM
DROPOUT = 0.5

model = models.FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM)
PATH = 'models/FastText.pt'

if args.model == 'RNN':
	INPUT_DIM = len(TEXT.vocab)
	EMBEDDING_DIM = 300
	HIDDEN_DIM = 256
	OUTPUT_DIM = 1

	model = models.RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
	PATH = 'models/RNN.pt'
elif args.model == 'LSTM':
	INPUT_DIM = len(TEXT.vocab)
	EMBEDDING_DIM = 300
	HIDDEN_DIM = 256
	OUTPUT_DIM = 1
	N_LAYERS = 2
	BIDIRECTIONAL = True
	DROPOUT = 0.5

	model = models.LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
	PATH = 'models/LSTM.pt'
elif args.model == 'FastText':
	INPUT_DIM = len(TEXT.vocab)
	EMBEDDING_DIM = 300
	OUTPUT_DIM = 1

	model = models.FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM)
	PATH = 'models/FastText.pt'
elif args.model == 'CNN':
	INPUT_DIM = len(TEXT.vocab)
	EMBEDDING_DIM = 300
	N_FILTERS = 100
	FILTER_SIZES = [3,4,5]
	OUTPUT_DIM = 1
	DROPOUT = 0.5

	model = models.CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
	PATH = 'models/CNN.pt'

if args.pre_train:
	print('Loading pre-trained model ...')
	pre_train = gensim.models.KeyedVectors.load_word2vec_format('sgns.literature.bigram')
	weights = torch.FloatTensor(pre_train.vectors)

	model.embedding.weight = torch.nn.Parameter(weights)
	print('Finish loading.')

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

def accuracy(preds, y):
	rounded_preds = torch.round(torch.sigmoid(preds))
	correct = (rounded_preds == y).float()
	acc = correct.sum() / len(correct)
	return acc

def train(model, iterator, optimizer, criterion):
	epoch_loss = 0
	epoch_acc = 0
	model.train()
	for batch in iterator:
		optimizer.zero_grad()
		predictions = model(batch.text).squeeze(1)
		loss = criterion(predictions, batch.label)
		acc = accuracy(predictions, batch.label)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc.item()
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
	epoch_loss = 0
	epoch_acc = 0
	model.eval()
	with torch.no_grad():
		for batch in iterator:
			predictions = model(batch.text).squeeze(1)
			loss = criterion(predictions, batch.label)
			acc = accuracy(predictions, batch.label)
			epoch_loss += loss.item()
			epoch_acc += acc.item()
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

if args.train:
	N_EPOCHS = 5

	print('Begin Training ...')
	for epoch in range(N_EPOCHS):
	    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
	    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
	    print('Epoch: ', epoch+1, \
	    	'Train Loss is:', train_loss, \
	    	'Train Acc is:', train_acc, \
	    	'Val Loss is:', valid_loss, \
	    	'Val Acc is: ', valid_acc
	    	)

	test_loss, test_acc = evaluate(model, test_iterator, criterion)
	print('Test Loss is:', test_loss, \
		'Test Acc is: ', test_acc)

	#torch.save(model.state_dict(), PATH)
	torch.save(model, PATH)
	print('Model saved.')

if args.predict:
	from spacy.lang.zh import Chinese
	import spacy
	nlp = Chinese()

	if os.path.exists(PATH):
		#model.load_state_dict(torch.load(PATH))
		model = torch.load(PATH)
		model.eval()
		print('Load model successfully.')
	else:
		print('No model available, please train first.')
		print('Now the prediction is random.')

	def predict_sentiment_CNN(sentence, min_len=5): # min_len is the largest filter size
	    tokenized = [tok.text for tok in nlp(sentence)]
	    if len(tokenized) < min_len:
	        tokenized += ['<pad>'] * (min_len - len(tokenized))
	    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
	    tensor = torch.LongTensor(indexed).to(device)
	    tensor = tensor.unsqueeze(1)
	    prediction = torch.sigmoid(model(tensor))
	    return prediction.item()

	def predict_sentiment(sentence):
	    tokenized = [tok.text for tok in nlp(sentence)]
	    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
	    tensor = torch.LongTensor(indexed).to(device)
	    tensor = tensor.unsqueeze(1)
	    prediction = torch.sigmoid(model(tensor))
	    return prediction.item()

	while True:
		sent = raw_input('Please input a sentence:')
		if args.model == 'CNN':
			print(predict_sentiment_CNN(sent))
		else:
			print(predict_sentiment(sent))

