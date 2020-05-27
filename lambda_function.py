"""
Copyright 2020, Sai Bhargav Dasari, All rights reserved
"""
try:
    import unzip_requirements
except ImportError:
    pass

import json
import pickle
import boto3
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


s3 = boto3.client('s3')
s3_res = boto3.resource('s3')
_, vocab_to_int, int_to_vocab, token_dict = pickle.loads(s3_res.Bucket("bhargav-ml-trained-models").Object("seinfeld_text_generation/preprocess.p").get()['Body'].read())
train_on_gpu = False

class RNN(nn.Module):
    def __init__(self, vocab_size=21388, output_size=21388, embedding_dim=250, hidden_dim=256, n_layers=2, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # Initialize layer sizes.
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim 
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Initialize Layers.
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # Run through embedding layer.
        batch_size = nn_input.shape[0]
        sequence_size = nn_input.shape[1]
        
        embeds = self.embed(nn_input)
        # LSTM layer
        lstm_output, hidden = self.lstm(embeds, hidden)
        # Turn lstm outputs into contiguous 
        lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)
        # Apply dropout
        lstm_dropout = self.dropout(lstm_output)
        # FC layer
        fc_out = self.fc(lstm_dropout)
        fc_out = fc_out.view(batch_size, sequence_size, self.output_size)
        final_layer = fc_out[:, -1 , :].squeeze(dim=1)
        # return one batch of output word scores and the hidden state    
        return final_layer, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden


def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, train_on_gpu=False, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    sequence_length = 10
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        current_seq = torch.LongTensor(current_seq)
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def lambda_handler(event, context):
    # TODO implement
    
    s3.download_file('bhargav-ml-trained-models', 'trained_state_dict.pt', '/tmp/trained_rnn.pt')
    trained_rnn = RNN()
    trained_rnn.load_state_dict(torch.load('/tmp/trained_rnn.pt'))
    trained_rnn.eval()
    gen_length = 300
    pad_word = '<PAD>'
    prime_word = event['body'].split('=')
    print(prime_word[-1])
    generated_script = generate(trained_rnn, vocab_to_int[prime_word[-1] + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
    
    return{ 
        'statusCode': 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body': generated_script
        }
