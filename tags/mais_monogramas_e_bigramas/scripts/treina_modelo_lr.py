# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:36:25 2016

@author: camposb
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from dados import carrega_dados

#Carrega dataset
#arquivo_dataset = 'D:\\Users\\camposb\\Projetos\\data\\dataset_enunciados_bigrams_nums.npy'
arquivo_dataset = 'D:\\Users\\camposb\\Projetos\\scripts\\dataset_enunciados.npy'
dataset = np.load(arquivo_dataset).item()


""" Área """
# Divide dataset em bases de treino e teste
X_train, X_test, y_train, y_test = train_test_split(dataset['X'], dataset['Área'], test_size=0.20, random_state=42)

# Traina um modelo usando regressão logística
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Determina acurácia nos dados de teste
print('Acurácia Área: %f' % lr.score(X_test, y_test))

# Obtém predições para os dados de teste
y_pred = lr.predict(X_test)

# Obtém nomes das áreas ordenado alfabeticamente
data_dir = 'D:\\Users\\camposb\\Projetos\\data' 
dados = carrega_dados(data_dir)
labels = sorted(np.unique(dados.Área))

# Determina matriz de confusão
cm = confusion_matrix(y_test, y_pred)

   
# Imprime e salva matriz de confusão
np.set_printoptions(precision=2, suppress=True)
#print(cm / cm.sum(1))
pd.DataFrame(cm, columns=labels).to_csv('cm.csv')

# Imprime precisão, revocação e f1 por classe
#print(classification_report(y_test, y_pred, target_names=labels))


# Imprime matriz de confusão com porcentagens
#np.set_printoptions(precision=2, suppress=True)
cm_por = cm / cm.sum(1)[:, None]
#print(cm_por)
pd.DataFrame(cm_por, columns=labels).to_csv('cm_porcentagens.csv')


""" Tema """
# Divide dataset em bases de treino e teste
X_train, X_test, y_train, y_test = train_test_split(dataset['X'], dataset['Tema'], test_size=0.20, random_state=42)

# Traina um modelo usando regressão logística
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Determina acurácia nos dados de teste
print('Acurácia Tema: %f' % lr.score(X_test, y_test))

# Obtém predições para os dados de teste
y_pred = lr.predict(X_test)



""" Subtema """
# Divide dataset em bases de treino e teste
X_train, X_test, y_train, y_test = train_test_split(dataset['X'], dataset['Subtema'], test_size=0.20, random_state=42)

# Traina um modelo usando regressão logística
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Determina acurácia nos dados de teste
print('Acurácia Subtema: %f' % lr.score(X_test, y_test))

# Obtém predições para os dados de teste
y_pred = lr.predict(X_test)