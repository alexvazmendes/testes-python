'''
Created on 23 de mai de 2018

@author: alexvaz
'''
import pandas as pd
import nltk
import random
import numpy as np

import tflearn
import tensorflow as tf


arquivo = pd.read_csv('textos.csv', encoding = 'utf-8')

stopwords = nltk.corpus.stopwords.words('portuguese')
stemmer = nltk.stem.RSLPStemmer()


palavras = []
docs = []
categorias = []

for index, row in arquivo.iterrows():
    frase = row['Texto'].lower()
    tokens = nltk.tokenize.word_tokenize(frase)
    
    tokens_validos = validas = [stemmer.stem(palavra) for palavra in tokens if palavra not in stopwords and len(palavra) > 2]
        
    palavras.extend(tokens_validos)
    
    categoria = row['Categoria']
    if categoria not in categorias:
        categorias.append(categoria)
    
    docs.append((tokens, row['Categoria']))
#    print row['Texto'], row['Categoria']

palavras = sorted(list(set(palavras)))

print palavras
print docs
print categorias

'''
textosPuros = arquivo['Texto']
frases = textosPuros.str.lower()
textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]


#nltk.download('punkt')
#nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

#nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()

dicionario = set()
for lista in textosQuebrados:
    validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
    dicionario.update(validas)
#print dicionario
'''

training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categorias)


for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in palavras:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categorias.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells
    # which catefory that bow belongs to.
    training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=5000, batch_size=8, show_metric=True)
model.save('model.tflearn')





