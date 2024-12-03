# Implementação de uma Rede Neural Artificial

## Descrição
Este projeto implementa uma Rede Neural Artificial (RNA) do zero em Python, utilizando conceitos básicos como:
- Estrutura de camadas com pesos e bias.
- Funções de ativação (Sigmoid, ReLU, Softmax).
- Algoritmo de retropropagação com gradiente descendente.
- Treinamento de modelos para tarefas de regressão, classificação binária e classificação multiclasse.

## Requisitos
- Python 3.8 ou superior.
- Dependências (instale com `pip install -r requirements.txt`):
  - numpy
  - matplotlib
  - scikit-learn
  - pandas

## Estrutura
- `src/`: Código-fonte da implementação.
- `notebooks/`: Fluxos documentados de treinamento e avaliação dos modelos.
- `data/`: Conjuntos de dados utilizados no projeto.

## Como Rodar

1. Clone o repositório:

``
git clone projeto
``

2. Instale as dependências:
``
pip install -r requirements.txt
``
3. Execute os notebooks no Jupyter:

``
jupyter notebook notebooks/
``

## Modelos Treinados
- **Regressão:** Predição contínua de valores com erro médio quadrático.
- **Classificação Binária:** Classificação de duas classes utilizando Sigmoid e ReLU.
- **Classificação Multiclasse:** Classificação de três ou mais classes utilizando Softmax.


## Referências de Dataset:
- **Regressão:** url
- **Classificação Binária:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **Classificação Multiclasse:** https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset


## Autores
- Acauã Pitta
- Isac
- Lucas Cunha