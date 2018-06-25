import os
import math
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from policy import Feature_Generator

def process_data():
    # Read Data
    with open("./shakespeare_processed.txt") as corpus_file:
        read_corpus = corpus_file.read().lower()
    print("Loaded a corpus of {0} characters".format(len(read_corpus)))
    
    # delate extra spaces and empty lines
    pre_data = []
    curr = ''
    for i in range(len(read_corpus)):
        if read_corpus[i] == '\n':
            if curr != '':
                pre_data.append(curr)
            pre_data.append('\n')
            curr = ''
        elif read_corpus[i] == ' ':
            if curr != '':
                pre_data.append(curr)
            curr = ''
        else:
            curr += read_corpus[i]
    if curr != '':
        pre_data.append(curr)
    
    data = []
    for i in range(len(pre_data)):
        if pre_data[i] == '\n' and len(data) > 0 and data[-1] == '\n':
            continue
        data.append(pre_data[i])
    
    # get unique words
    words = sorted(list(set(data)))
    num_words = len(words)
    encoding = {c: i for i, c in enumerate(words)}
    decoding = {i: c for i, c in enumerate(words)}
    print("Our corpus contains {0} unique characters.".format(num_words))

    # get preprocessed word feature vectors
    with open('vectors.txt', 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]
    
    ind_vectors = {}
    for word in words:
        if word == '\n':
            ind_vectors[encoding[word]] = [0.0 for _ in range(len(vectors['the']))]
        else:
            ind_vectors[encoding[word]] = vectors[word]
    
    # Let every '\n' be a start of a new sequence
    starts = [i for i in range(len(data)) if data[i] == '\n']
    
    sentence_length = 25
    X_data = []
    y_data = []
    for i in starts:
        if i + sentence_length >= len(data):
            break
        sentence = data[i:i+sentence_length]
        next_words = data[i+1:i + sentence_length+1]
        X_data.append([encoding[word] for word in sentence])
        y_data.append([encoding[word] for word in next_words])
    
    num_sentences = len(X_data)
    print("Sliced our corpus into {0} sentences of length {1}".format(num_sentences, sentence_length))    

    # vectorize x
    print("Vectorizing X...")
    X = np.zeros((num_sentences, sentence_length, len(vectors['the'])), dtype=np.float)
    for i, sentence in enumerate(X_data):
        for t, encoded_word in enumerate(sentence):
            X[i, t] = ind_vectors[encoded_word]
    
    num_sentences = len(X_data)
    print("Sliced our corpus into {0} sentences of length {1}".format(num_sentences, sentence_length))

    return X, np.asarray(y_data), ind_vectors, encoding, decoding

def save_model(model, path=None):
    assert path is not None
    print("saving")
    if use_cuda:
        model = model.cpu()
    pickle.dump((model, ), open(path, 'wb'))
    if use_cuda:
        model = model.cuda()

X, y, vectors, encoding, decoding = process_data()
tot_data = X.shape[0]
train_X = X[:int(tot_data * 0.9)]
train_y = y[:int(tot_data * 0.9)]
test_X = X[int(tot_data * 0.9):]
test_y = y[int(tot_data * 0.9):]
print(train_X.shape, train_y.shape, len(encoding))
use_cuda = True
generator = Feature_Generator(len(vectors[0]), 1024, vectors, len(encoding), use_cuda, layers = 3, dropout = 0.8).double()

####
#generator, = pickle.load(open('word_feature_model.p', 'rb'))
####

gen_criterion = nn.NLLLoss()
gen_optimizer = optim.Adam(generator.parameters())
if use_cuda:
    generator = generator.cuda()
    gen_criterion = gen_criterion.cuda()

tot_epochs = 20
batch_size = 32
data_size = train_X.shape[0]
seq_length = train_X.shape[1]
for epoch in range(tot_epochs):
    ind = np.random.permutation(data_size)
    X = train_X[ind]
    y = train_y[ind]
    total_loss = []

    # training model
    for i in range(data_size // batch_size):
        batch_X = Variable(torch.from_numpy(X[batch_size * i : batch_size * (i + 1)]).double())
        batch_y = Variable(torch.from_numpy(y[batch_size * i : batch_size * (i + 1)]).long()).contiguous().view(-1)
        if use_cuda:
            batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
        
        pred = generator.forward(batch_X)
        loss = gen_criterion(pred, batch_y)
        total_loss.append(loss.data[0])
        
        gen_optimizer.zero_grad()
        loss.backward()
        gen_optimizer.step()
        
    print("training loss:", np.mean(total_loss))
    
    # validation
    val_loss = []
    for i in range(test_X.shape[0] // batch_size):
        val_X = Variable(torch.from_numpy(test_X[batch_size * i : batch_size * (i + 1)]).double(), volatile=True)
        val_y = Variable(torch.from_numpy(test_y[batch_size * i : batch_size * (i + 1)]).long(), volatile=True).contiguous().view(-1)
        if use_cuda:
            val_X, val_y = val_X.cuda(), val_y.cuda()
        pred_val = generator.forward(val_X)
        loss_val = gen_criterion(pred_val, val_y)
        val_loss.append(loss_val.data[0])
    print("validation loss:", np.mean(val_loss))

#save_model(generator, 'word_feature_model.p')

# generate poems
seed = []
seed_sentence = ['shall', 'i', 'compare', 'thee', 'to', "summer's", 'day', '\n' ]
for i in seed_sentence:
    seed.append(encoding[i])

sample = generator.sample(500, encoding['\n'], 1.0, seed)
output = ''
print(sample)
for d in sample:
    output += (' ' + decoding[d])
print(output)