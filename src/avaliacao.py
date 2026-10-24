"""Funções de avaliação de classificadores: métricas completas, relatório de validação, limiar, custo e ganho.

Classe de interesse = 1 (funcionário saiu da empresa)."""
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from src.graficos import salvar

ROTULOS = ['Ficou (0)', 'Saiu (1)']


def calcular_metricas_completas(y_true, y_pred, y_proba):
    """Todas as métricas de um classificador binário (classe de interesse = 1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred, zero_division=0),
        'Recall (Sensibilidade)': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, y_proba),
        'PR-AUC': average_precision_score(y_true, y_proba),
        'Especificidade': tn / (tn + fp) if (tn + fp) else 0,
        'Verdadeiros Positivos': tp, 'Verdadeiros Negativos': tn,
        'Falsos Positivos': fp, 'Falsos Negativos': fn,
    }


def relatorio_validacao(nome, modelo, X_treino, y_treino, X_val, y_val, limiar=0.5, nome_arquivo=None):
    """Avalia um modelo já treinado na validação, no formato padrão do projeto.

    Mostra: amostra gabarito x probabilidade x previsão, métricas, PR-AUC no treino x validação (sobreajuste),
    relatório de classificação e matriz de confusão. Devolve (métricas, probabilidades da validação).
    """
    proba = modelo.predict_proba(X_val)[:, 1]
    previsto = (proba >= limiar).astype(int)

    print('AMOSTRA: GABARITO x PROBABILIDADE x PREVISÃO (limiar = {:.2f})'.format(limiar))
    display(pd.DataFrame({'Gabarito': np.asarray(y_val), 'Probabilidade de sair': proba.round(3),
                          'Previsão da máquina': previsto}).head(8))

    metricas = calcular_metricas_completas(y_val, previsto, proba)
    print('=' * 50)
    print(f'MÉTRICAS DE VALIDAÇÃO: {nome}')
    print('=' * 50)
    for chave in ['Acurácia', 'Precisão', 'Recall (Sensibilidade)', 'F1-Score', 'AUC-ROC', 'PR-AUC']:
        print(f'{chave:<24} {metricas[chave]:.4f}')
    print(f'{"":<24} (sorteio: PR-AUC = {np.mean(y_val):.4f}, AUC-ROC = 0,50)')

    proba_treino = modelo.predict_proba(X_treino)[:, 1]
    pr_auc_treino = average_precision_score(y_treino, proba_treino)
    print('\nSOBREAJUSTE (treino x validação):')
    print(f'  PR-AUC: treino {pr_auc_treino:.4f} | validação {metricas["PR-AUC"]:.4f}'
          f' | diferença {pr_auc_treino - metricas["PR-AUC"]:+.4f}')
    metricas['PR-AUC treino'] = pr_auc_treino

    print('\n' + classification_report(y_val, previsto, target_names=ROTULOS, digits=3, zero_division=0))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_val, previsto, display_labels=['Ficou', 'Saiu'],
                                            cmap='Blues', colorbar=False, ax=ax)
    ax.set(xlabel='Previsto pelo modelo', ylabel='Real')
    ax.set_title(f'Matriz de confusão: {nome}', fontsize=12, loc='left')
    if nome_arquivo:
        salvar(nome_arquivo)
    plt.show()
    return metricas, proba


def plot_comparacao_modelos(metricas, taxa_positivos, nome_arquivo):
    """Painel com uma barra por modelo em cada métrica e a linha de referência de um "modelo que chuta"."""
    modelos = list(metricas)
    cores = sns.color_palette('Set2', len(modelos))
    paineis = [('Acurácia', 1 - taxa_positivos, "sempre 'fica'"), ('Precisão', taxa_positivos, 'sorteio'),
               ('Recall (Sensibilidade)', None, None), ('F1-Score', None, None), ('AUC-ROC', 0.5, 'sorteio'),
               ('PR-AUC', taxa_positivos, 'sorteio')]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (metrica, referencia, rotulo) in zip(axes.ravel(), paineis):
        valores = [metricas[m][metrica] for m in modelos]
        ax.bar(modelos, valores, color=cores, edgecolor='black', alpha=0.85)
        for i, v in enumerate(valores):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
        if referencia is not None:
            ax.axhline(referencia, color='gray', ls='--', lw=1.2, label=f'{rotulo} = {referencia:.2f}')
            ax.legend(loc='upper right', fontsize=9)
        ax.set_title(metrica, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', labelsize=9, rotation=20)
    fig.suptitle('Comparação dos modelos na validação (limiar 0,50)', fontsize=16, fontweight='bold', x=0.01, ha='left')
    plt.tight_layout()
    salvar(nome_arquivo)
    plt.show()


def limiar_otimo_f1(y, proba):
    """Limiar que maximiza o F1 (o 0,5 padrão raramente é o melhor corte operacional)."""
    precisao, revocacao, limiares = precision_recall_curve(y, proba)
    f1 = np.divide(2 * precisao * revocacao, precisao + revocacao, out=np.zeros_like(precisao),
                   where=(precisao + revocacao) > 0)
    i = int(np.argmax(f1[:-1]))
    return float(limiares[i]), float(f1[i])


def curva_custo(y, proba, custo_perda, custo_abordagem, limiares=np.round(np.arange(0.01, 1.0, 0.01), 2)):
    """Custo de cada limiar: cada saída não prevista custa a reposição (`custo_perda`) e cada pessoa
    abordada pelo RH (saia ou não) custa uma ação de retenção (`custo_abordagem`).

    Premissa simplificadora: a abordagem evita a saída (o custo de quem é abordado e sai mesmo assim não entra).
    """
    y, proba = np.asarray(y), np.asarray(proba)
    linhas = []
    for limiar in limiares:
        abordado = proba >= limiar
        perdidas = int(((~abordado) & (y == 1)).sum())
        linhas.append({'Limiar': limiar, 'Abordados': int(abordado.sum()),
                       'Saídas previstas': int((abordado & (y == 1)).sum()), 'Saídas não previstas': perdidas,
                       'Abordagens desnecessárias': int((abordado & (y == 0)).sum()),
                       'Custo das saídas não previstas': perdidas * custo_perda,
                       'Custo das abordagens': abordado.sum() * custo_abordagem})
    tabela = pd.DataFrame(linhas)
    tabela['Custo total'] = tabela['Custo das saídas não previstas'] + tabela['Custo das abordagens']
    return tabela


def resumo_custo(y, abordado, custo_perda, custo_abordagem):
    """Resultado de uma regra de decisão em dinheiro, comparado a não ter modelo (ninguém é abordado)."""
    y, abordado = np.asarray(y), np.asarray(abordado).astype(bool)
    sem_modelo = (y == 1).sum() * custo_perda
    com_modelo = ((~abordado) & (y == 1)).sum() * custo_perda + abordado.sum() * custo_abordagem
    return {'Custo sem modelo': float(sem_modelo), 'Custo com o modelo': float(com_modelo),
            'Economia': float(sem_modelo - com_modelo), 'Economia (%)': float(1 - com_modelo / sem_modelo),
            'Abordados': int(abordado.sum()), 'Saídas previstas': int((abordado & (y == 1)).sum()),
            'Saídas no grupo': int((y == 1).sum())}


def ganho_vs_sorteio(y, proba):
    """Quantas saídas são encontradas abordando as N pessoas de maior risco, contra a média de um sorteio."""
    y = np.asarray(y)[np.argsort(-np.asarray(proba))]
    abordados = np.arange(1, len(y) + 1)
    return pd.DataFrame({'Abordados': abordados, 'Modelo': np.cumsum(y), 'Sorteio': abordados * y.mean()})


def bootstrap_metricas(y, proba, limiar, n=1000, semente=42):
    """Intervalo de 95% (bootstrap estratificado) da PR-AUC, do recall e da precisão no limiar escolhido."""
    y, proba = np.asarray(y), np.asarray(proba)
    rng = np.random.default_rng(semente)
    positivos, negativos = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    linhas = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(positivos, len(positivos)), rng.choice(negativos, len(negativos))])
        previsto = proba[idx] >= limiar
        linhas.append({'PR-AUC': average_precision_score(y[idx], proba[idx]),
                       'Recall': recall_score(y[idx], previsto, zero_division=0),
                       'Precisão': precision_score(y[idx], previsto, zero_division=0)})
    return pd.DataFrame(linhas).quantile([0.025, 0.5, 0.975]).T.rename(columns={0.025: 'IC 2,5%', 0.5: 'mediana',
                                                                              0.975: 'IC 97,5%'})


def salvar_ficha_modelo(caminho, **campos):
    """Grava a ficha do modelo (JSON): tudo o que é preciso saber para usá-lo corretamente."""
    ficha = {'data_treino': datetime.now().strftime('%Y-%m-%d %H:%M'), **campos}
    with open(caminho, 'w', encoding='utf-8') as arquivo:
        json.dump(ficha, arquivo, ensure_ascii=False, indent=2, default=float)
    return ficha


def explicar_previsao(modelo, entrada, referencia, n=3):
    """Variáveis que mais afastam a chance de sair de um funcionário da chance do funcionário de referência.

    Vale para o Pipeline (pré-processamento + Regressão Logística): a log-chance é a soma de coeficiente × valor
    transformado, então a diferença para a referência se divide exatamente entre as variáveis (as colunas do
    one-hot voltam a ser a variável original). Mostra associação no modelo, não causa.
    Retorna as n de maior efeito, com o multiplicador da chance (> 1 aumenta o risco, < 1 diminui).
    """
    preprocessador, regressao = modelo.named_steps['prep'], modelo[-1]
    diferenca = (preprocessador.transform(entrada) - preprocessador.transform(referencia))[0] * regressao.coef_[0]
    origem = []
    for _, transformador, colunas in preprocessador.transformers_:
        if transformador == 'drop':
            continue
        if hasattr(transformador, 'categories_'):   # uma coluna do one-hot por categoria
            origem += [coluna for coluna, categorias in zip(colunas, transformador.categories_) for _ in categorias]
        else:
            origem += list(colunas)
    efeito = pd.Series(diferenca, index=origem).groupby(level=0, sort=False).sum()
    principais = efeito[efeito.abs().sort_values(ascending=False).index[:n]]
    return pd.DataFrame({'variavel': principais.index, 'efeito_log_chance': principais.values,
                         'multiplicador': np.exp(principais.values)})
