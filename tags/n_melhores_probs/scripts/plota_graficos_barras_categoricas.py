# -*- coding: utf-8 -*-
"""
Plota gráficos de barras mostrando quantitativos das variáveis categóricas do dataset de enunciados.
"""

import matplotlib.pyplot as plt
from dados import carrega_dados


# Carrega os dados
data_dir = 'D:\\Users\\camposb\\Projetos\\data' 
dados = carrega_dados(data_dir)

# Plota gráfico de barras mostrando quantidade de exemplares por Área
plt.figure()
dados.Área.value_counts().plot(kind='bar')
plt.grid()
plt.title('Número de Exemplares por Área')
plt.xlabel('Área')
plt.ylabel('Quantidade')
plt.subplots_adjust(bottom=0.35)
for p in plt.gca().patches:
    plt.gca().annotate('%3.d'%int(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005 + 25))

# Plota gráfico de barras mostrando quantidade de exemplares por Tema
plt.figure()
dados.Tema.value_counts().plot(kind='bar')
plt.grid()
plt.title('Número de Exemplares por Tema')
plt.xlabel('Tema')
plt.ylabel('Quantidade')
plt.subplots_adjust(bottom=0.35)
for tick in plt.gca().xaxis.get_major_ticks(): tick.label.set_fontsize(7)

# Plota gráfico de barras mostrando quantidade de exemplares por Subtema
plt.figure()
dados.Subtema.value_counts().plot(kind='bar')
plt.grid()
plt.title('Número de Exemplares por Subtema')
plt.xlabel('Subtema')
plt.ylabel('Quantidade')
plt.subplots_adjust(bottom=0.45)

# Imprime números de categorias distintas
print('#Áreas: %s' %len(dados.Área.unique()))
print('#Temas: %s' %len(dados.Tema.unique()))
print('#Subtemas: %s' %len(dados.Subtema.unique()))


