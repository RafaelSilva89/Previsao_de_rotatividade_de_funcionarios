"""Teste do app Streamlit: abre sem erro, mostra a probabilidade e reage ao perfil do funcionário."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def abrir():
    return AppTest.from_file(APP, default_timeout=60).run()


def probabilidade(app):
    return int(app.metric[0].value.replace('%', ''))


def test_app_abre_e_mostra_a_probabilidade():
    app = abrir()
    assert not app.exception
    assert app.metric[0].label == 'Probabilidade de sair'
    assert 0 <= probabilidade(app) <= 100


def test_hora_extra_aumenta_o_risco():
    app = abrir()
    app.sidebar.toggle[0].set_value(False).run()
    sem = probabilidade(app)
    app.sidebar.toggle[0].set_value(True).run()
    com = probabilidade(app)
    assert not app.exception
    assert com > sem


def test_lista_do_rh_encontra_mais_que_o_acaso():
    app = abrir()   # o RH aborda 60 pessoas (valor inicial do slider)
    encontradas = int(app.metric[3].value.split(' de ')[0])
    acaso = float(app.metric[4].value)
    assert encontradas > acaso


def test_explica_o_que_mais_pesou():
    app = abrir()
    app.sidebar.toggle[0].set_value(True).run()   # só a hora extra muda em relação ao funcionário típico
    assert not app.exception
    explicacao = [item.value for item in app.markdown if 'chance de sair' in item.value]
    assert len(explicacao) == 1 and 'Hora extra' in explicacao[0] and 'multiplica' in explicacao[0]
