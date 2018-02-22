# -*- coding: utf-8 -*-
"""
É preciso fazer o donwload dos seguintes pacotes do NLTK utilizando nltk.download():
 + stopwords (palavras frequentes)
 + rslp (removedor de sufixos (stemmer))
 
 TODO:
 - Salvar em cache a lista de palavras frequentes e o índice.
"""

import pandas as pd
import numpy as np
import nltk
from dados import carrega_dados
  

def canonicaliza_texto(texto):
    """
    Retorna o texto canonicalizado:
     - Converte tudo para caixa baixa
     - Remove sinais de pontuação
     - Remove dígitos
    """
    
    # Converte texto para caixa baixa
    texto = texto.lower()
        
    # Remove sinais de pontuação
    sinais_de_pontuacao = '?!,.(){}[]\';:\\""//º-'
    for sinal in sinais_de_pontuacao:
        texto = texto.replace(sinal,' ')
        
    # Remove dígitos
    digitos = '0123456789'
    for digito in digitos:
        texto = texto.replace(digito,'')
    
    return texto

def remove_stopwords(corpus):
    """
    Retorna corpus sem as palavras comuns ('stopwords').
    """
    
    # Obtém lista de palavras comuns
    stopwords = nltk.corpus.stopwords.words('portuguese')
    
    # Obtém lista de todas as palavras ("trimando" quando necessário)
    palavras = [palavra.strip() for palavra in corpus.split() if palavra not in stopwords]
    
    return ' '.join(palavras)


def remove_sufixos(corpus):
    """
    Retorna corpus com as palavras sem os sufixos (stemização)
    """
    
    # Separa palavras em lista
    palavras = corpus.split()    
    
    # Remove sufixos das palavras
    stemmer = nltk.stem.RSLPStemmer()
    raizes_palavras = [stemmer.stem(palavra) for palavra in palavras]
    
    return ' '.join(raizes_palavras)


def determina_palavras_frequentes(corpus, frequencia_minima=5):
    """
    Retorna Series com as palavras cuja frequência de ocorrência no corpus seja maior que a mínima.
    """
    
    # Calcula frequência de ocorrência de cada radical
    frequencias_das_palavras = pd.Series(corpus.split()).value_counts()
    
    # Dtermina palavras comuns
    palavras_comuns = frequencias_das_palavras[frequencias_das_palavras > frequencia_minima]
    
    return palavras_comuns.index.values
    
def determina_bigramas_frequentes(corpus, frequencia_minima=10):
    """
    TODO:
    """
    
    # Determina frequencia de cada bigrama
    bi = pd.Series(list(nltk.bigrams(corpus.split()))).value_counts()
    
    # Retorna birgramas com frequência acima da mínima
    return bi[bi > frequencia_minima]


def cria_mapa_item_para_indice(lista):
    """
    Cria mapa associando a cada palavra a respectiva posição dentro do vetor de características
    """
    return {w:i for i, w in enumerate(lista)}

        
def mapeia_categorias_para_codigos(categorias):
    """
    Recebe a coluna 'Áreas' do dataset como entrada e retorna uma lista com os códigos referentes às 
    respectivas áreas.
    
    """
    
    # Dicionário com itens {área:código}
    mapa = {a:i for i, a in enumerate(sorted(categorias.unique())) }
    
    return [mapa[area] for area in categorias]


def extrai_caracteristicas(enunciado, palavras_frequentes):
    """
    Retorna o vetor de características 
    
    """
    
    # Cria mapa associando a cada palavra a respectiva posição dentro do vetor de características
    indice_monograma = cria_mapa_item_para_indice(palavras_frequentes)

    
    # Inicializa vetor de características com zeros
    caracteristicas = len(palavras_frequentes)*[0]
    
    # Pré-processa texto
    enunciado = remove_sufixos(remove_stopwords(canonicaliza_texto(enunciado)))    
    
    # Para cada palavra no enunciado
    for w in enunciado.split():
        
        # Caso a palavra seja frequente        
        if w.strip() in palavras_frequentes:
            
            # Seta o valor 1 para a respectiva posição no vetor de características
            indice = indice_monograma[w]
            caracteristicas[indice] = 1
        
    return np.array(caracteristicas)
    

# Carrega os dados
data_dir = 'D:\\Users\\camposb\\Projetos\\data' 
dados = carrega_dados(data_dir)

# Obtém string com o texto de todos os enunciados concatenados
corpus = ''.join(dados.Enunciado.values)



# Canonicaliza texto
corpus = canonicaliza_texto(corpus)

# Remove palavras comuns
corpus = remove_stopwords(corpus)

# Remove sufixos das palavras
corpus = remove_sufixos(corpus)

# Inicializa lista vazia para adicionar os vetores de características
vetores = []

# Obtém lista de palavras frequentes no corpus
palavras_frequentes = determina_palavras_frequentes(corpus)

# Para cada enunciado
for i, enunciado in enumerate(dados.Enunciado):
    print("Processando enunciado %d de %d" % (i, len(dados.Enunciado)))
    
    # Cria o vetor de características
    v = extrai_caracteristicas(enunciado, palavras_frequentes)    
    vetores.append(v)


# Salva dataset no arquivo
dataset = {}
dataset['X'] = np.array(vetores)
dataset['Área'] = mapeia_categorias_para_codigos(dados.Área)
dataset['Tema'] = mapeia_categorias_para_codigos(dados.Área + ' - ' + dados.Tema)
dataset['Subtema'] = mapeia_categorias_para_codigos(dados.Área + ' - ' + dados.Tema + ' - ' + dados.Subtema)
#dataset['Áreas'] = dados.Área.values
np.save('dataset_enunciados', dataset)

