#Imports
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from pprint import pprint
from time import time


#Global Variables
MAX_LEN = 50
SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5
use_cuda = torch.cuda.is_available() and True

#Classes
class LanguageModel:
    def __init__(self, name):
        self.name = name
        self.w2i_map = {}
        self.w2f_map = {}
        self.i2w_map = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2

    def load_sentence_to_lang_model(self, sentence):
        for word in sentence:
            self.add_word_to_lang_model(word)

    def add_word_to_lang_model(self, word):
        if word not in self.w2i_map:
            self.w2i_map[word] = self.n_words
            self.w2f_map[word] = 1
            self.i2w_map[self.n_words] = word
            self.n_words += 1
        else:
            self.w2f_map[word] += 1

class Encoder(nn.Module):
    def __init__(self, in_vocab_size, in_embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = in_vocab_size
        self.embedding_dim = in_embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(in_vocab_size, in_embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim)

    def forward(self, in_word, in_hidden):
        embedded_word = self.embedding(in_word).view(1, 1, -1)
        gru_out, out_hidden = self.gru(embedded_word, in_hidden)
        return gru_out, out_hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Decoder(nn.Module):
    def __init__(self, out_vocab_size, out_embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = out_embedding_dim
        self.output_dim = out_vocab_size
        self.embedding = nn.Embedding(out_vocab_size, out_embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, in_word, in_hidden):
        embedded_word = self.embedding(in_word).view(1, 1, -1)
        gru_out, out_hidden = self.gru(embedded_word, in_hidden)
        out_word = self.log_softmax(self.out(gru_out[0]))
        return out_word, out_hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

#Functions
def normalize_word(word):
    word = word.lower().strip()
    word = re.sub(r"([.!?}])", r" \1", word)
    # word = re.sub(r"[^a-zA-Z.!?]+", r" ", word)
    return word

def filter_pairs(pairs, max_len=1e4):
    return [pair for pair in pairs if len(pair[0])<max_len and len(pair[1])<max_len]

def load_lang_model_from_file(pos_fname, neg_fname):
    print(f'Reading Lines from file: {pos_fname}')
    lines = open(pos_fname, encoding='utf-8').read().strip().split('\n')
    pos_pairs = [([normalize_word(word) for word in line.split(' ')], ['positive']) for line in lines]
    print(f'Loaded {len(pos_pairs)} lines')
    print(f'Reading Lines from file: {neg_fname}')
    lines = open(neg_fname, encoding='utf-8').read().strip().split('\n')
    neg_pairs = [([normalize_word(word) for word in line.split(' ')], ['negative']) for line in lines]
    print(f'Loaded {len(neg_pairs)} lines')
    pairs = pos_pairs
    pairs.extend(neg_pairs)
    random.shuffle(pairs)
    print(f'Total lines: {len(pairs)}')
    sentences_lang = LanguageModel('sentences')
    sentiments_lang = LanguageModel('sentiments')
    pairs = filter_pairs(pairs, max_len=MAX_LEN)
    print(f'Filtering the pairs based on MAX_LEN: {MAX_LEN}')
    print(f'Selected {len(pairs)} sentences pairs after filtering')
    print(f'Counting words...')
    for pair in pairs:
        sentences_lang.load_sentence_to_lang_model(pair[0])
        sentiments_lang.load_sentence_to_lang_model(pair[1])
    print(f'Vocab size of <{sentences_lang.name}> (including SOS, EOS): {sentences_lang.n_words}\n'
          f'Vocab size of <{sentiments_lang.name}> (including SOS, EOS): {sentiments_lang.n_words}')
    return sentences_lang, sentiments_lang, pairs

def train(src_variable, tgt_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_len=MAX_LEN):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    src_len = src_variable.size()[0]
    tgt_len = tgt_variable.size()[0]
    # encoder_outputs = Variable(torch.zeros(max_len, encoder.hidden_dim))
    # encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(src_len):
        encoder_output, encoder_hidden = encoder(src_variable[ei], encoder_hidden)
        # encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        #Teacher forcing: use the target as the next input
        for di in range(tgt_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, tgt_variable[di])
            decoder_input = tgt_variable[di] #Teacher forcing
    else:
        #No teacher forcing: use its own predictions as the next input
        for di in range(tgt_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            loss += criterion(decoder_output, tgt_variable[di])
            if ni == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0]/tgt_len

def indexes_from_sentence(lang, sentence):
    return [lang.w2i_map[word] for word in sentence]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variables_from_pair(pair):
    src_variable = variable_from_sentence(sentences_lang, pair[0])
    tgt_variable = variable_from_sentence(sentiments_lang, pair[1])
    return (src_variable, tgt_variable)

def train_iters(encoder, decoder, n_iters, lrate=0.01, print_every=10):
    t0 = time()
    losses = []
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=lrate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=lrate)
    training_pairs = [variables_from_pair(random.choice(train_pairs)) for _ in range(n_iters)]
    criterion = nn.NLLLoss()
    for i in range(1, n_iters+1):
        training_pair = training_pairs[i-1]
        src_variable = training_pair[0]
        tgt_variable = training_pair[1]
        loss = train(src_variable, tgt_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        losses.append(loss)
        if i%print_every == 0:
            print(f'Loss at iteration {i}/{n_iters}: {sum(losses)/len(losses)}. Took {time()-t0} secs')
            losses = []
            t0 = time()
    pass

def testing(encoder, decoder, sentence, max_len=MAX_LEN):
    src_variable = variable_from_sentence(sentences_lang, sentence)
    src_len = src_variable.size()[0]
    encoder_hidden = encoder.init_hidden()
    encoder_outputs = Variable(torch.zeros(max_len, encoder.hidden_dim))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    for ei in range(src_len):
        encoder_output, encoder_hidden = encoder(src_variable[ei], encoder_hidden)
        # encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden
    decoded_words = []
    # decoder_attentions = torch.zeros(max_len, max_len)
    for di in range(max_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(sentiments_lang.i2w_map[ni])
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    return decoded_words

def test_random(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(test_pairs)
        print('>', ' '.join(pair[0]))
        print('=', pair[1][0])
        output_words = testing(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
    pass

def score(encoder, decoder, pairs):
    num_correct = 0
    total_sents = len(pairs)
    for pair in pairs:
        output = testing(encoder, decoder, pair[0])
        output = output[0]
        if output == pair[1][0]:
            num_correct += 1
    return num_correct/total_sents
#Main
if __name__ == '__main__':
    pos_fname = 'rt-polarity.pos'
    neg_fname = 'rt-polarity.neg'
    sentences_lang, sentiments_lang, pairs = load_lang_model_from_file(pos_fname, neg_fname)
    pprint(random.choice(pairs))
    test_fraction = 0.1
    test_num_pairs = int(len(pairs) * test_fraction)
    train_pairs = pairs[:-test_num_pairs]
    test_pairs = pairs[-test_num_pairs:]
    print(f'{len(train_pairs)} training sentences, {len(test_pairs)} testing sentences')
    embedding_size = 100
    hidden_size = 100
    enc = Encoder(sentences_lang.n_words, embedding_size, hidden_size)
    dec = Decoder(sentiments_lang.n_words, embedding_size, hidden_size)
    if use_cuda:
        print('GPU Available. Using GPU!!! :)')
        enc = enc.cuda()
        dec = dec.cuda()
    else:
        print('Using CPU :(')
    num_epochs = 5
    num_iters = len(train_pairs)*5
    train_iters(enc, dec, num_iters, lrate=0.2, print_every=1000)
    print(f'Training Accuracy: {score(enc, dec, train_pairs)}')
    print(f'Testing Accuracy: {score(enc, dec, test_pairs)}')
    test_random(enc, dec)
    pass