'''
Created on 23 de mai de 2018

@author: alexvaz
'''
import pandas as pd
import nltk
arquivo = pd.read_csv('textos.csv', encoding = 'utf-8')

for index, row in arquivo.iterrows():
    print row['Texto'], row['Categoria']


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