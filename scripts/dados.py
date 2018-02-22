# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import os
     
def listFiles(directory, extension):
    """
    List recursively all files with the specified extension in the informed directory
    """
    files=[]
  
    for dirPath, dirNames, fileNames in os.walk(directory):         #@UnusedVariable
        for fileName in fileNames:
            baseName, fileExtension = os.path.splitext(fileName)    #@UnusedVariable
             
            if(fileExtension == '.'+extension):
                files.append(os.path.join(dirPath, fileName))
  
    return files


def carrega_dados(diretorio_dados):
    """
    Retorna dataset dos enunciados.
    
    Carrega dados arquivos xls com enunciados contidos no diretório especificado,
    remove colunas sem importância, remove espaços dos nomes das colunas e retorna 
    dataset compilado.
    """
    
    # Carrega lista de arquivos de dados no diretório especificado
    arquivos = listFiles(diretorio_dados,'xls')
    
    # Cria dataset como concatenação de todos os arquivos de dados
    dados = pd.DataFrame()
    for a in arquivos:
        d = pd.read_excel(a)
        dados = dados.append(d, ignore_index=True)
    
    # Remove espaços dos nomes das colunas
    dados.columns = [ x.replace(' ', '') for x in dados.columns]
    
    # Remove colunas que não possuem nenhuma informação relevante
    #dados.drop('Tipo', inplace=True, axis=1)
    #dados.drop('Acórdão', inplace=True, axis=1)
    #dados.drop('Assunto', inplace=True, axis=1)
    
    return dados


def carrega_corpus(diretorio_dados):
    """
    Retorna uma string contendo o texto de todos os enunciados.
    """
    
    # Carrega dataset dos enunciados
    dados = carrega_dados(diretorio_dados)

    # Obtém string com o texto de todos os enunciados concatenados
    corpus = ''.join(dados.Enunciado.values)
    
    return corpus

