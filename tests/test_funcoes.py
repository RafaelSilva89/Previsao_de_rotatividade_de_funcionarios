"""Testes das funções de src/: leitura e preparo da base, métricas, custo, ganho e testes estatísticos."""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.avaliacao import (bootstrap_metricas, calcular_metricas_completas, curva_custo, explicar_previsao,
                           ganho_vs_sorteio, limiar_otimo_f1, resumo_custo)
from src.config import ARQUIVO_BASE, CATEGORICAS, FEATURES, SEM_INFORMACAO, TAXAS
from src.dados import carregar_base, colunas_sem_informacao, criar_variaveis, funcionario_para_modelo, preparar_base
from src.estatistica import odds_ratio, teste_mann_whitney, teste_qui_quadrado
from src.monitoramento import psi, relatorio_drift


@pytest.fixture(scope='module')
def base():
    return carregar_base(ARQUIVO_BASE)


def test_carregar_base_traduz_e_converte(base):
    assert base.shape == (1470, 35)
    assert set(base['saiu'].unique()) == {0, 1}
    assert set(base['hora_extra'].unique()) == {0, 1}
    assert base['saiu'].sum() == 237
    assert not base[CATEGORICAS].isna().any(axis=None)   # todo valor de texto tem tradução
    assert 'Solteiro(a)' in set(base['estado_civil'])


def test_colunas_sem_informacao(base):
    assert set(colunas_sem_informacao(base)) == {'contagem', 'maior_18', 'horas_padrao'}


def test_preparar_base_remove_e_cria_colunas():
    df = preparar_base(ARQUIVO_BASE)
    assert not set(SEM_INFORMACAO) & set(df.columns)
    assert set(FEATURES) <= set(df.columns)
    assert not set(TAXAS) & set(FEATURES)
    assert 'genero' not in FEATURES   # atributo sensível fica fora do modelo
    assert df['satisfacao_media'].between(1, 4).all()
    assert df['fracao_carreira_na_empresa'].between(0, 1).all()


def test_criar_variaveis_valores_conhecidos():
    linha = pd.DataFrame([dict(satisfacao_ambiente=1, satisfacao_trabalho=2, satisfacao_relacionamento=3, equilibrio_vida=4,
                               anos_experiencia=10, empresas_anteriores=1, anos_empresa=5)])
    criada = criar_variaveis(linha).iloc[0]
    assert criada['satisfacao_media'] == pytest.approx(2.5)
    assert criada['anos_por_empresa'] == pytest.approx(5)
    assert criada['fracao_carreira_na_empresa'] == pytest.approx(0.5)
    sem_experiencia = criar_variaveis(linha.assign(anos_experiencia=0, anos_empresa=0)).iloc[0]
    assert sem_experiencia['fracao_carreira_na_empresa'] == 1


def test_funcionario_para_modelo(base):
    funcionario = preparar_base(ARQUIVO_BASE).iloc[0].to_dict()
    entrada = funcionario_para_modelo(funcionario)
    assert entrada.columns.tolist() == FEATURES and len(entrada) == 1


def test_metricas_completas():
    metricas = calcular_metricas_completas([1, 1, 0, 0], [1, 0, 1, 0], [0.9, 0.4, 0.6, 0.1])
    assert metricas['Precisão'] == pytest.approx(0.5)
    assert metricas['Recall (Sensibilidade)'] == pytest.approx(0.5)
    assert metricas['Verdadeiros Positivos'] == 1 and metricas['Falsos Negativos'] == 1


def test_curva_e_resumo_de_custo():
    y = np.array([1, 1, 0, 0, 0])
    proba = np.array([0.9, 0.3, 0.8, 0.2, 0.1])
    tabela = curva_custo(y, proba, custo_perda=100, custo_abordagem=20, limiares=[0.5])
    linha = tabela.iloc[0]
    assert linha['Abordados'] == 2 and linha['Saídas não previstas'] == 1
    assert linha['Custo total'] == pytest.approx(1 * 100 + 2 * 20)
    resumo = resumo_custo(y, proba >= 0.5, 100, 20)
    assert resumo['Custo sem modelo'] == 200 and resumo['Economia'] == pytest.approx(60)


def test_limiar_otimo_f1_separa_classes_perfeitas():
    limiar, f1 = limiar_otimo_f1([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert f1 == pytest.approx(1) and 0.2 < limiar <= 0.8


def test_ganho_vs_sorteio():
    ganho = ganho_vs_sorteio([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert ganho['Modelo'].tolist() == [1, 2, 2, 2]
    assert ganho['Sorteio'].iloc[-1] == pytest.approx(2)


def test_bootstrap_metricas_intervalo_contem_mediana():
    gerador = np.random.default_rng(0)
    y = np.r_[np.ones(50), np.zeros(250)].astype(int)
    proba = np.clip(y * 0.4 + gerador.uniform(0, 0.6, len(y)), 0, 1)
    intervalos = bootstrap_metricas(y, proba, 0.5, n=200)
    assert (intervalos['IC 2,5%'] <= intervalos['mediana']).all() and (intervalos['mediana'] <= intervalos['IC 97,5%']).all()


def test_testes_estatisticos():
    condicao = pd.Series([1] * 50 + [0] * 50)
    alvo = pd.Series([1] * 30 + [0] * 20 + [1] * 10 + [0] * 40)
    assert odds_ratio(condicao == 1, alvo) == pytest.approx((30 * 40) / (20 * 10))
    assert teste_qui_quadrado(condicao, alvo)['tamanho do efeito'] > 0.3
    assert teste_mann_whitney([5, 6, 7, 8], [1, 2, 3, 4])['tamanho do efeito'] == pytest.approx(1)


def test_explicar_previsao_divide_a_diferenca_entre_as_variaveis():
    df = pd.DataFrame({'cor': ['a', 'b', 'c', 'a', 'b', 'c'] * 5, 'x': np.arange(30.0), 'y': [0, 1] * 15})
    prep = ColumnTransformer([('categorias', OneHotEncoder(), ['cor']), ('numericas', StandardScaler(), ['x'])])
    modelo = Pipeline([('prep', prep), ('modelo', LogisticRegression())]).fit(df[['cor', 'x']], df['y'])
    referencia, entrada = df[['cor', 'x']].iloc[[0]], pd.DataFrame({'cor': ['c'], 'x': [25.0]})
    explicacao = explicar_previsao(modelo, entrada, referencia, n=5)
    assert sorted(explicacao['variavel']) == ['cor', 'x']   # o one-hot volta a ser uma variável só
    diferenca = modelo.decision_function(entrada)[0] - modelo.decision_function(referencia)[0]
    assert explicacao['efeito_log_chance'].sum() == pytest.approx(diferenca)
    assert np.allclose(explicacao['multiplicador'], np.exp(explicacao['efeito_log_chance']))
    assert explicar_previsao(modelo, entrada, referencia, n=1)['variavel'].tolist() == [explicacao['variavel'].iloc[0]]


# --- Monitoramento (data drift) ---

def test_psi_igual_perto_de_zero_e_deslocado_severo():
    rng = np.random.default_rng(0)
    referencia = rng.normal(0, 1, 5000)
    assert psi(referencia, rng.normal(0, 1, 5000)) < 0.02
    assert psi(referencia, rng.normal(1, 1, 5000)) > 0.25


def test_psi_variavel_binaria():
    assert psi(np.r_[np.zeros(700), np.ones(300)], np.r_[np.zeros(700), np.ones(300)]) < 0.001
    assert psi(np.r_[np.zeros(700), np.ones(300)], np.r_[np.zeros(300), np.ones(700)]) > 0.25


def test_relatorio_drift_aponta_a_coluna_que_mudou():
    rng = np.random.default_rng(1)
    referencia = pd.DataFrame({'a': rng.normal(0, 1, 2000), 'b': rng.normal(0, 1, 2000)})
    novo = pd.DataFrame({'a': rng.normal(0, 1, 2000), 'b': rng.normal(2, 1, 2000)})
    relatorio = relatorio_drift(referencia, novo, ['a', 'b'])
    assert relatorio.index[0] == 'b'
    assert relatorio.loc['b', 'Status (PSI)'] == 'drift severo: retreinar'
    assert relatorio.loc['a', 'Status (PSI)'] == 'estável'
    assert bool(relatorio.loc['b', 'K-S indica mudança'])
