'''
@author: alexvaz
'''
import pandas as pd
import nltk
import random
import numpy as np

import tflearn
import tensorflow as tf

stopwords = nltk.corpus.stopwords.words('portuguese')
stemmer = nltk.stem.RSLPStemmer()

def get_tokens_validos(frase):
    frase = frase.lower()
    tokens = nltk.tokenize.word_tokenize(frase)
    tokens_validos = [stemmer.stem(palavra) for palavra in tokens if palavra not in stopwords and len(palavra) > 2]
    return tokens_validos

def get_vetor_teste(frase):
    global dicionario
    
    tokens_frase = get_tokens_validos(frase)
   
    vetor = [0]*len(dicionario)
    for s in tokens_frase:
        for i, w in enumerate(dicionario):
            if w == s:
                vetor[i] = 1

    return(np.array(vetor))

dicionario = []
docs = []
categorias = []

arquivo = pd.read_csv('textos.csv', encoding = 'utf-8')
for index, row in arquivo.iterrows():
    tokens_validos = get_tokens_validos(row['Texto'])
        
    dicionario.extend(tokens_validos)
    
    categoria = row['Categoria']
    if categoria not in categorias:
        categorias.append(categoria)
    
    docs.append((tokens_validos, row['Categoria']))

dicionario = sorted(list(set(dicionario)))

dados_treino = []
saida_vazia = [0] * len(categorias)

for doc in docs:
    vetor_binario = []
    tokens = doc[0]
    
    for p in dicionario:
        vetor_binario.append(1) if p in tokens else vetor_binario.append(0)

    saida = list(saida_vazia)
    saida[categorias.index(doc[1])] = 1
    
    dados_treino.append([vetor_binario, saida])



random.shuffle(dados_treino)
dados_treino = np.array(dados_treino)
print dados_treino

treino_x = list(dados_treino[:, 0])
treino_y = list(dados_treino[:, 1])

tf.reset_default_graph()

rede = tflearn.input_data(shape=[None, len(treino_x[0])])
rede = tflearn.fully_connected(rede, 10)
rede = tflearn.fully_connected(rede, 10)
rede = tflearn.fully_connected(rede, len(treino_y[0]), activation='softmax')
rede = tflearn.regression(rede)


modelo = tflearn.DNN(rede, tensorboard_dir='tflearn_logs')
modelo.fit(treino_x, treino_y, n_epoch=1000, batch_size=5, show_metric=True)
modelo.save('model.tflearn')

# TESTES
arquivo2 = pd.read_csv('testes.csv', encoding = 'utf-8')

num_testes = 0
num_acertos = 0
for index, row in arquivo2.iterrows():
    previsao = categorias[np.argmax(modelo.predict([get_vetor_teste(row['Texto'])]))]
    categoria_teste = row['Categoria']
    num_testes = num_testes + 1
    print "Frase {0}, Previsao: {1}, Categoria Real: {2}".format(num_testes, previsao, categoria_teste)
    if previsao == categoria_teste:
        num_acertos = num_acertos + 1

print "Total de testes: {0}".format(num_testes)
print "Acertos: {0}".format(num_acertos)
taxa_de_acerto = 100.0 * num_acertos / num_testes
print "Taxa de Acertos: {0}%".format(taxa_de_acerto)