# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:52:03 2016

@author: camposb
"""

import pandas as pd
import numpy as np
import nltk
from dados import carrega_dados
  



def canonicaliza_texto(texto):
    """
    Retorna o texto canonicalizado:
     - Converte tudo para caixa baixa
     - Remove sinais de pontuação (ou substitui por espaços)
    """
    
    # Converte texto para caixa baixa
    texto = texto.lower()
        
    # Remove sinais de pontuação
    sinais_de_pontuacao = '?!,.(){}[]\';:"º%'
    for sinal in sinais_de_pontuacao:
        texto = texto.replace(sinal,'')
    
    # Substitui por espaços
    sinais_de_pontuacao = '\\//-'
    for sinal in sinais_de_pontuacao:
        texto = texto.replace(sinal,' ')        

    return texto

def mapeia_categorias_para_codigos(categorias):
    """
    Recebe a coluna 'Áreas' do dataset como entrada e retorna uma lista com os códigos referentes às 
    respectivas áreas.
    
    """
    
    # Dicionário com itens {área:código}
    mapa = {a:i for i, a in enumerate(sorted(categorias.unique())) }
    
    return [mapa[area] for area in categorias]




# Carrega os dados
data_dir = 'D:\\Users\\camposb\\Projetos\\data' 
dados = carrega_dados(data_dir)

# Obtém dicionário mapeando cada palavra ao respectivo vetor
imersoes = np.load('..\\w2e-embeddings\\types-features.npy')
vocabulario = open('..\\w2e-embeddings\\vocabulary.txt', encoding='utf8').read().split('\n')
mapa_vetores = { w:imersoes[i] for i,w in enumerate(vocabulario) }  

#
enunciados = [canonicaliza_texto(x) for x in dados.Enunciado.values]

# Inicializa lista vazia para adicionar os vetores de características
vetores = []

# Para cada enunciado
for i, enunciado in enumerate(enunciados):
    print("Processando enunciado %d de %d" % (i+1, len(dados.Enunciado)))

    v = np.zeros((1, 50))[0]
    n = 0.
    
    #
    palavras = [w.strip() for w in enunciado]
    for w in palavras:
        if w in mapa_vetores:
            v += mapa_vetores[w]            
            n += 1.
        else:
            #print(w)
            pass
   
    # 
    v /= n
    #print(len(enunciado), n, n/len(enunciado))
    vetores.append(v)   
    
# Salva dataset no arquivo
dataset = {}
dataset['X'] = np.array(vetores)
dataset['Área'] = mapeia_categorias_para_codigos(dados.Área)
dataset['Tema'] = mapeia_categorias_para_codigos(dados.Área + ' - ' + dados.Tema)
dataset['Subtema'] = mapeia_categorias_para_codigos(dados.Área + ' - ' + dados.Tema + ' - ' + dados.Subtema)
#dataset['Áreas'] = dados.Área.values
np.save('dataset_enunciados', dataset)
    
    
    