"""Monitoramento do modelo em produção: data drift pelo PSI e pelo teste de Kolmogorov-Smirnov."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Faixas usuais do PSI (indústria financeira)
PSI_MODERADO = 0.10
PSI_SEVERO = 0.25


def psi(esperado, atual, bins=10):
    """Population Stability Index: o quanto a distribuição `atual` se afastou da `esperado` (referência).

    Os limites das faixas vêm dos percentis da referência. Variáveis com poucos valores (ex.: 0/1, 1 a 4)
    geram percentis repetidos, que são unidos. PSI < 0,10: estável; 0,10 a 0,25: moderado; > 0,25: severo.
    """
    esperado, atual = np.asarray(esperado, dtype=float), np.asarray(atual, dtype=float)
    limites = np.unique(np.percentile(esperado, np.linspace(0, 100, bins + 1)))
    if len(limites) < 3:  # variável praticamente constante na referência
        limites = np.unique(np.concatenate([limites, [limites[0] - 0.5, limites[-1] + 0.5]]))
    limites[0], limites[-1] = -np.inf, np.inf
    pct_esperado = np.histogram(esperado, limites)[0] / len(esperado)
    pct_atual = np.histogram(atual, limites)[0] / len(atual)
    pct_esperado, pct_atual = np.clip(pct_esperado, 0.001, None), np.clip(pct_atual, 0.001, None)  # evita log(0)
    return float(np.sum((pct_atual - pct_esperado) * np.log(pct_atual / pct_esperado)))


def status_psi(valor):
    if valor < PSI_MODERADO:
        return 'estável'
    return 'drift moderado' if valor < PSI_SEVERO else 'drift severo: retreinar'


def relatorio_drift(referencia, novo, colunas, alfa=0.05):
    """PSI e teste K-S de cada coluna numérica, comparando os dados novos com os de referência (treino)."""
    linhas = {}
    for coluna in colunas:
        estatistica, p_valor = ks_2samp(referencia[coluna], novo[coluna])
        valor_psi = psi(referencia[coluna], novo[coluna])
        linhas[coluna] = {'PSI': valor_psi, 'Status (PSI)': status_psi(valor_psi), 'K-S': estatistica,
                          'p-valor (K-S)': p_valor, 'K-S indica mudança': p_valor < alfa}
    return pd.DataFrame.from_dict(linhas, orient='index').sort_values('PSI', ascending=False)
