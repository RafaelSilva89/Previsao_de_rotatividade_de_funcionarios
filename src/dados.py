"""Leitura, tradução e preparo da base de RH (IBM HR Analytics Attrition)."""
import numpy as np
import pandas as pd

from src.config import COLUNAS, FEATURES, SEM_INFORMACAO, VALORES


def carregar_base(caminho):
    """Lê o CSV, traduz colunas e valores para português e converte `saiu` e `hora_extra` em 1/0."""
    df = pd.read_csv(caminho, encoding='utf-8-sig').rename(columns=COLUNAS)
    for coluna, traducao in VALORES.items():
        df[coluna] = df[coluna].map(traducao)
    df['saiu'] = (df['saiu'] == 'Yes').astype(int)
    df['hora_extra'] = (df['hora_extra'] == 'Yes').astype(int)
    return df


def colunas_sem_informacao(df):
    """Colunas com um único valor em toda a base (não ajudam a diferenciar ninguém)."""
    return [coluna for coluna in df.columns if df[coluna].nunique() == 1]


def criar_variaveis(df):
    """Variáveis criadas a partir das originais (nenhuma usa o alvo).

    - satisfacao_media: média das 4 satisfações (ambiente, trabalho, relacionamento, equilíbrio), de 1 a 4;
    - anos_por_empresa: anos de experiência ÷ (empresas anteriores + 1), estabilidade na carreira;
    - fracao_carreira_na_empresa: anos na empresa ÷ anos de experiência (1 = carreira toda aqui).
    """
    df = df.copy()
    df['satisfacao_media'] = df[['satisfacao_ambiente', 'satisfacao_trabalho', 'satisfacao_relacionamento',
                                 'equilibrio_vida']].mean(axis=1)
    df['anos_por_empresa'] = df['anos_experiencia'] / (df['empresas_anteriores'] + 1)
    df['fracao_carreira_na_empresa'] = np.where(df['anos_experiencia'] > 0,
                                                df['anos_empresa'] / df['anos_experiencia'].clip(lower=1), 1.0)
    return df


def preparar_base(caminho):
    """Base pronta para a análise: traduzida, sem as colunas sem informação e com as variáveis criadas."""
    return criar_variaveis(carregar_base(caminho).drop(columns=SEM_INFORMACAO))


def funcionario_para_modelo(funcionario):
    """Converte o dicionário de um funcionário (como o app recebe) para a tabela de entrada do modelo."""
    return criar_variaveis(pd.DataFrame([funcionario]))[FEATURES]
