# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:36:25 2016

@author: camposb
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


#Carrega dataset
arquivo_dataset = 'D:\\Users\\camposb\\Projetos\\data\\dataset_enunciados_bigrams_nums.npy'
dataset = np.load(arquivo_dataset).item()

# Inicializa listas para guardas as acurácias para cada uam das variáveis que se deseja prever
acuracias = {'Área':[], 'Tema':[], 'Subtema':[]}

# Para cada uma das variáveis que se deseja predizer
for categoria in ['Área','Tema','Subtema']:

    # Divide dataset em bases de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(dataset['X'], dataset[categoria], test_size=0.20, random_state=42)
    
    # Traina um modelo usando regressão logística
    lr = LogisticRegression()
    lr.fit(X_train, y_train)    
      
    # Probabilidades de pertencemento a cada classe
    probs = lr.predict_proba(X_test)
    
    # Número máximo 
    NUM_MAX = 10
    
    # Para o número de máximas probabilidade variando de 1 a NUM_MAX
    for num in range(1, NUM_MAX+1):
    
        # Determina índices das classes de maior probabilidade
        highest_probs = probs.argsort()[:, ::-1][:, 0:num]
        
        # Corrige "bug" do sklearn. Os índices retornadas por 'predict_proba' são os índices
        # de lr.classes_ e não os índices "normais". Na maior parte dos casos, estes seriam os mesmo,
        # mas no caso de não haver todas as classes nos dados de teste passados para 'predict_proba',
        # os índices são alterados (pois há furos na ordem dos índices).
        highest_probs = lr.classes_[highest_probs]
        
        # Inicializa a quantidade de acertos com zero
        acertos = 0
        
        # Para cada classe (dos enunciados de teste)
        for i, clazz in enumerate(y_test):
        
            # Incrementa o número de acertos caso a classe do enunciado esteja entre as NUM classes mais prováveis
            if clazz in highest_probs[i, :]:    
                acertos += 1
        
        # Adiciona a acurácia à lista de acurácias
        acuracia = acertos/len(y_test)
        acuracias[categoria].append(acuracia)
        
    
    #xxx   
    print(acuracias[categoria])
    plt.plot(range(1, NUM_MAX+1), acuracias[categoria], marker='o', label=categoria)
    
        
plt.grid()       
plt.ylabel('Acurácia (%)')
plt.xlabel('N')
plt.title('Acurácia obtida ao se considerar as N probabildiades mais altas')
plt.legend(loc='lower right')
