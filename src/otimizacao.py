"""Comparação de métodos de ajuste de hiperparâmetros: Grid Search, Random Search e Bayesiana (Optuna)."""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_validate


def objetivo_cv(pipeline, espaco, X, y, cv, scoring, n_jobs=1):
    """Cria a função objetivo do Optuna: sugere hiperparâmetros, roda a validação cruzada e devolve a média.

    `espaco` é uma função que recebe o trial e devolve o dicionário de hiperparâmetros (nomes do Pipeline,
    ex.: 'modelo__C'). O desvio e o score de treino ficam guardados no trial, para a comparação com o sklearn.
    """
    def objetivo(trial):
        modelo = clone(pipeline).set_params(**espaco(trial))
        r = cross_validate(modelo, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=True)
        trial.set_user_attr('desvio', float(r['test_score'].std()))
        trial.set_user_attr('treino', float(r['train_score'].mean()))
        return float(r['test_score'].mean())
    return objetivo


def resumo_sklearn(nome, busca, tempo, sinal=1):
    """Linha da tabela de comparação a partir de um GridSearchCV/RandomizedSearchCV (com return_train_score=True).

    `sinal=-1` desfaz o sinal negativo das métricas de erro do sklearn (ex.: neg_mean_absolute_error -> MAE).
    """
    i = busca.best_index_
    r = busca.cv_results_
    return {'Método': nome, 'Melhor score (CV)': sinal * r['mean_test_score'][i], 'Desvio entre dobras': r['std_test_score'][i],
            'Score no treino': sinal * r['mean_train_score'][i], 'Combinações testadas': len(r['params']),
            'Modelos treinados': len(r['params']) * busca.n_splits_, 'Tempo (s)': tempo,
            'Melhores hiperparâmetros': {k.replace('modelo__', ''): v for k, v in busca.best_params_.items()}}


def resumo_optuna(nome, study, tempo, n_splits, sinal=1):
    """Linha da tabela de comparação a partir de um estudo do Optuna criado com `objetivo_cv`."""
    melhor = study.best_trial
    return {'Método': nome, 'Melhor score (CV)': sinal * melhor.value, 'Desvio entre dobras': melhor.user_attrs['desvio'],
            'Score no treino': sinal * melhor.user_attrs['treino'], 'Combinações testadas': len(study.trials),
            'Modelos treinados': len(study.trials) * n_splits, 'Tempo (s)': tempo,
            'Melhores hiperparâmetros': {k.replace('modelo__', ''): v for k, v in melhor.params.items()}}


def historico_optuna(study, maior_melhor=True):
    """Score de cada trial e o melhor score acumulado até ele (para o gráfico do histórico da otimização)."""
    valores = np.array([t.value for t in study.trials], dtype=float)
    acumulado = np.maximum.accumulate(valores) if maior_melhor else np.minimum.accumulate(valores)
    return valores, acumulado
