# -*- coding: utf-8 -*-
"""
Plota gráfico de dispesão com componentes principais do dataset de enunciados.

"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from dados import carrega_dados



def plota_scatter_com_legenda_de_cor(x, y, valores):
    """ 
    Cria um gráfico de dispersão com pontos em x,y e cores corresopndentes às classes
    da variável valores e com as legendas (label) com os nomes das classes.
    """
    
    # Obtém lista de classes
    classes = sorted(np.unique(valores))
    
    # Cria mapa de cores com o número de cores igual ao númer de classes
    colors = plt.cm.get_cmap('gist_rainbow', len(classes))

    # Para cada classe
    for i, classe in enumerate(classes):
        # Filtra pontos da classe
        xc = x[valores == classe]
        yc = y[valores == classe]
        
        # Plota dados da classes com a cor e o label (que aparece ao se usar legend) correta
        plt.scatter(xc, yc, color=colors(len(xc)*[i]), s=10, label=classe)


#Carrega dataset
arquivo_dataset = 'D:\\Users\\camposb\\Projetos\\scripts\\dataset_enunciados.npy'
dataset = np.load(arquivo_dataset).item()

# Converte lista de códigos das áreas para lista de nomes das áreas
data_dir = 'D:\\Users\\camposb\\Projetos\\data' 
dados = carrega_dados(data_dir)
nomes_areas = sorted(np.unique(dados.Área))
areas = np.array([nomes_areas[codigo] for codigo in dataset['Área']])

# Extrai compoenentes princiapsi com PCA
pca = PCA(n_components=2)
componentes_principais = pca.fit_transform(dataset['X'])

# Atribui às variáveis x e y os dois principais componenetes extraídos pelo PCA (coluna 1 e coluna 2)
x = componentes_principais[:, 0]
y = componentes_principais[:, 1]

    
# Plota os pontos com as cores certas de acordo com o agrupamento
plota_scatter_com_legenda_de_cor(x, y, areas)
plt.grid()
plt.legend(fontsize=8)
plt.title('Componentes principais (PCA) dos enunciados')



'''
Código para validar as imersões.

# Obtém dicionário mapeando cada palavra ao respectivo vetor
imersoes = np.load('..\\w2e-embeddings\\types-features.npy')

# Extrai compoenentes princiapsi com PCA
pca = PCA(n_components=2)
imersoes = pca.fit_transform(imersoes)

vocabulario = open('..\\w2e-embeddings\\vocabulary.txt', encoding='utf8').read().split('\n')
mapa_vetores = { w:imersoes[i] for i,w in enumerate(vocabulario) }  



for pais in ['brasil','belgica','chile','frança','mexico','alemanha','argentina']:
    plt.scatter(*mapa_vetores[pais], color='red', s=50)
    
for animal in ['boi','vaca','zebra','cavalo','ovelha','tartaruga','jabuti','hiena','jacaré','raposa']:
    plt.scatter(*mapa_vetores[animal], color='green', s=50)    
    
    
for animal in ['eu','tu','ele','nós','vós','eles','me','te','lhe','nos','vos','lhes']:
    plt.scatter(*mapa_vetores[animal], color='blue', s=50)    
    
for animal in ['um','dois','três','quatro','cinco','seis','sete','oito','nove','dez','zero']:
    plt.scatter(*mapa_vetores[animal], color='yellow', s=50)   
plt.grid()


for pais in ['alemanha', 'frança']:
    plt.scatter(*mapa_vetores[pais], color='purple', s=50)
'''