import random
import numpy as np
from glob import glob
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

# bool value for running the model on caracters or words
run_on_words = False # choose False for running on characters

# Loading corpus_file
corpus_file = open("beatles_corpus.txt", "r")
data = corpus_file.read()
if run_on_words : # for running on words
    data = data.split()
token_list=data
unique_tokens_list = set(token_list)
print "Unique corpus tokens: "+str(len(unique_tokens_list))
corpus_file.close()
# Parameters to run the script :
epochs = 60  # no epochs
batch_size_value = 128 #  values could be from 120 to len(token_list) - choosing 128 as default for running while on cpu
max_len = 20 # set a fixed vector size so that we can look at specific windows of characters

# defining the model on keras
model_rnn = Sequential()
model_rnn.add(LSTM(512, return_sequences=True, input_shape=(max_len, len(unique_tokens_list))))
model_rnn.add(Dropout(0.2))
model_rnn.add(LSTM(512, return_sequences=False))
model_rnn.add(Dropout(0.2))
model_rnn.add(Dense(len(unique_tokens_list)))
model_rnn.add(Activation('softmax'))
model_rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop')
step = 1
inputs = []
outputs = []
for i in range(0, len(token_list) - max_len, step):
    inputs.append(token_list[i:i+max_len])
    outputs.append(token_list[i+max_len])


print inputs

print outputs

token_labels = {ch:i for i, ch in enumerate(unique_tokens_list)}
labels_token = {i:ch for i, ch in enumerate(unique_tokens_list)}


example = 'cab dab'
example_char_labels = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    ' ' : 4
}

# using bool to reduce memory usage
X = np.zeros((len(inputs), max_len, len(unique_tokens_list)), dtype=np.bool)
y = np.zeros((len(inputs), len(unique_tokens_list)), dtype=np.bool)

# set the appropriate indices to 1 in each one-hot vector
for i, example in enumerate(inputs):
    for t, token in enumerate(example):
        X[i, t, token_labels[token]] = 1
    y[i, token_labels[outputs[i]]] = 1




model_rnn.fit(X, y, batch_size=len(inputs), nb_epoch=epochs)

def generate(temperature=0.35, seed=None, predicate=lambda x: len(x) < 100):
    if seed is not None and len(seed) < max_len:
        raise Exception('Seed text must be at least {} chars long'.format(max_len))

    else:
        start_idx = random.randint(0, len(text) - max_len - 1)
        seed = text[start_idx:start_idx + max_len]

    sentence = seed
    generated = sentence

    while predicate(generated):
        # generate the input tensor
        # from the last max_len characters generated so far
        x = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_labels[char]] = 1.

        # this produces a probability distribution over characters
        probs = model_rnn.predict(x, verbose=0)[0]

        # sample the character to use based on the predicted probabilities
        next_idx = sample(probs, temperature)
        next_char = labels_char[next_idx]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

def sample(probs, temperature):
    """samples an index from a vector of probabilities
    (this is not the most efficient way but is more robust)"""
    a = np.log(probs)/temperature
    dist = np.exp(a)/np.sum(np.exp(a))
    choices = range(len(probs))
    return np.random.choice(choices, p=dist)

for i in range(epochs):
    print('epoch %d'%i)
    # set nb_epoch to 1 since we're iterating manually
    # comment this out if you just want to generate text
    model_rnn.fit(X, y, batch_size=batch_size_value, nb_epoch=1)

    # preview
    for temp in [0.2, 0.5, 1., 1.2]:
        print('temperature: %0.2f'%temp)
        print('%s'%generate(temperature=temp))

