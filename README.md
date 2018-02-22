# Modelo de Classificação de Enunciados de Jurisprudência

Projeto criado no final de 2015 para classificação de enunciados de jurisprudência em Área, Tema e Subtema.

No código há alguns exemplos de:
* Criação de modelos preditivos diversos usando `scikitlearn`.
* Redução de dimencionalidade utilizando **PCA**.
* Pré-processamento e conversão de texto para **bag-of-word** e **vetores de palavras** usando `NLTK` e `word2vec` (vetores treinados por [grupo da USP](http://nilc.icmc.usp.br/nlpnet/training.html)).
* Criação de gráficos simples com `matplotlib` para realiação de nálise exploratório.
* Tarefas comuns de machine learning, utilando o `scikitlearn`.

## Observações
* As versões mais antigas do projeto estão salvas na pasta ``tags``.
* Os arquivos de dados (gerados a partir dos enunciados pelos scripts `cria_datasets.py`) foram excluídos do repositório (através da inclusão da extensão `.npy` no .gitignore) pois eles eram muito grandes.

__________________________

# Atualização:

Modelo em atualização para utilização de word embeddings com LSTM, buscando melhoria na performance do modelo preditivo. Arquivos novos no diretório nnmodel. 

Embeddings utilizados de: http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc#