"""App Streamlit: risco de saída de um funcionário e lista de prioridade para o RH."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.append(str(RAIZ))

from src.avaliacao import explicar_previsao  # noqa: E402
from src.config import CUSTO_REPOSICAO, NOMES_LEGIVEIS, VALORES  # noqa: E402
from src.dados import funcionario_para_modelo  # noqa: E402

st.set_page_config(page_title='Rotatividade de funcionários', page_icon='👥', layout='wide')


@st.cache_resource
def carregar_modelo():
    pacote = joblib.load(RAIZ / 'Model' / 'modelo_rotatividade.pkl')
    ficha = json.loads((RAIZ / 'Model' / 'modelo_rotatividade.json').read_text(encoding='utf-8'))
    return pacote, ficha


@st.cache_data
def carregar_teste():
    return pd.read_csv(RAIZ / 'reports' / 'previsoes_teste.csv')


def dolares(valor):
    return 'US$ ' + f'{valor:,.0f}'.replace(',', '.')


def formatar(coluna, valor):
    if coluna == 'hora_extra':
        return 'sim' if valor else 'não'
    if coluna == 'salario_mensal':
        return dolares(valor)
    if isinstance(valor, str) or float(valor).is_integer():
        return str(valor if isinstance(valor, str) else int(valor))
    return f'{valor:.2f}'.replace('.', ',')


pacote, ficha = carregar_modelo()
teste = carregar_teste()
padrao = pacote['padrao']   # valor típico (mediana ou mais comum) de cada coluna, para o que não for informado
limiar = pacote['limiar']

st.title('👥 Quem está em risco de sair?')
st.caption('Modelo treinado com a base fictícia IBM HR Analytics (1.470 funcionários, 16% saíram). '
           'A probabilidade indica prioridade para uma conversa do RH, não uma certeza.')

aba_funcionario, aba_lista, aba_modelo = st.tabs(['🧑 Funcionário', '📋 Lista de prioridade do RH', 'ℹ️ Sobre o modelo'])

with aba_funcionario:
    with st.sidebar:
        st.header('Dados do funcionário')
        idade = st.slider('Idade', 18, 60, int(padrao['idade']))
        salario = st.number_input('Salário mensal (US$)', 1000, 20000, int(padrao['salario_mensal']), step=100)
        hora_extra = st.toggle('Faz hora extra', value=bool(padrao['hora_extra']))
        estado_civil = st.selectbox('Estado civil', list(VALORES['estado_civil'].values()),
                                    index=list(VALORES['estado_civil'].values()).index(padrao['estado_civil']))
        viagens = st.selectbox('Viagens a trabalho', list(VALORES['viagens'].values()),
                               index=list(VALORES['viagens'].values()).index(padrao['viagens']))
        departamento = st.selectbox('Departamento', list(VALORES['departamento'].values()),
                                    index=list(VALORES['departamento'].values()).index(padrao['departamento']))
        cargo = st.selectbox('Cargo', list(VALORES['cargo'].values()),
                             index=list(VALORES['cargo'].values()).index(padrao['cargo']))
        nivel_cargo = st.slider('Nível do cargo (1 a 5)', 1, 5, int(padrao['nivel_cargo']))
        nivel_acoes = st.slider('Nível de ações da empresa (0 a 3)', 0, 3, int(padrao['nivel_acoes']))
        anos_empresa = st.slider('Anos na empresa', 0, 40, int(padrao['anos_empresa']))
        anos_experiencia = st.slider('Anos de experiência (total)', 0, 40, max(int(padrao['anos_experiencia']), anos_empresa))
        satisfacao = st.slider('Satisfação com o trabalho (1 a 4)', 1, 4, int(padrao['satisfacao_trabalho']))
        ambiente = st.slider('Satisfação com o ambiente (1 a 4)', 1, 4, int(padrao['satisfacao_ambiente']))
        equilibrio = st.slider('Equilíbrio vida-trabalho (1 a 4)', 1, 4, int(padrao['equilibrio_vida']))

    funcionario = dict(padrao)
    funcionario.update(idade=idade, salario_mensal=salario, hora_extra=int(hora_extra), estado_civil=estado_civil,
                       viagens=viagens, departamento=departamento, cargo=cargo, nivel_cargo=nivel_cargo,
                       nivel_acoes=nivel_acoes, anos_empresa=anos_empresa, anos_experiencia=max(anos_experiencia, anos_empresa),
                       satisfacao_trabalho=satisfacao, satisfacao_ambiente=ambiente, equilibrio_vida=equilibrio,
                       anos_cargo_atual=min(int(padrao['anos_cargo_atual']), anos_empresa),
                       anos_gestor_atual=min(int(padrao['anos_gestor_atual']), anos_empresa),
                       anos_ultima_promocao=min(int(padrao['anos_ultima_promocao']), anos_empresa))
    entrada = funcionario_para_modelo(funcionario)[pacote['features']]
    probabilidade = float(pacote['modelo'].predict_proba(entrada)[0, 1])

    col1, col2, col3 = st.columns(3)
    col1.metric('Probabilidade de sair', f'{probabilidade:.0%}')
    col2.metric('Taxa média da empresa', f"{pacote['taxa_base']:.0%}",
                delta=f'{probabilidade / pacote["taxa_base"]:.1f}x a média', delta_color='off')
    col3.metric('Recomendação', '🔴 Conversar' if probabilidade >= limiar else '🟢 Acompanhar')
    st.caption(f'O RH é acionado a partir de {limiar:.0%} de probabilidade (limiar de menor custo na validação, '
               f'com reposição de {dolares(CUSTO_REPOSICAO)} = {pacote["proporcao_custo"]}x o custo de uma abordagem). '
               'Os campos não mostrados usam o valor típico da empresa.')

    figura = go.Figure(go.Indicator(mode='gauge+number', value=probabilidade * 100, number={'suffix': '%'},
                                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#C0392B'},
                                           'threshold': {'line': {'color': 'black', 'width': 3}, 'value': limiar * 100}}))
    figura.update_layout(height=260, margin=dict(t=20, b=10))
    st.plotly_chart(figura, use_container_width=True)

    st.subheader('O que mais pesou para este funcionário')
    referencia = funcionario_para_modelo(padrao)[pacote['features']]
    explicacao = explicar_previsao(pacote['modelo'], entrada, referencia)
    explicacao = explicacao[explicacao['efeito_log_chance'].abs() > 0.01]   # ignora efeitos desprezíveis
    if explicacao.empty:
        st.write('Este perfil é igual ao funcionário típico da empresa: nenhuma variável se destaca.')
    for linha in explicacao.itertuples():
        sentido = '🔺 multiplica' if linha.multiplicador > 1 else '🔻 divide'
        fator = f'{max(linha.multiplicador, 1 / linha.multiplicador):.1f}'.replace('.', ',')
        st.markdown(f'- **{NOMES_LEGIVEIS[linha.variavel]}**: {formatar(linha.variavel, entrada[linha.variavel].iloc[0])} '
                    f'(típico: {formatar(linha.variavel, referencia[linha.variavel].iloc[0])}) → {sentido} a chance de sair '
                    f'por {fator}')
    st.caption('Comparação com o funcionário típico da empresa (valor mais comum de cada campo), que tem '
               f"{pacote['modelo'].predict_proba(referencia)[0, 1]:.0%} de probabilidade de sair. "
               'Mostra o que o modelo associa ao risco, não a causa: serve para orientar a conversa do RH.')

with aba_lista:
    st.subheader('Quantas saídas o RH encontra abordando as pessoas de maior risco?')
    st.caption(f'Funcionários do teste ({len(teste)} pessoas que o modelo nunca viu, {int(teste["real"].sum())} saíram), '
               'ordenados pela probabilidade de sair.')
    vagas = st.slider('Quantas pessoas o RH consegue abordar?', 5, len(teste), 60, step=5)
    ordenado = teste.sort_values('probabilidade', ascending=False)
    encontradas = int(ordenado['real'].head(vagas).sum())
    sorteio = vagas * teste['real'].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric('Saídas encontradas pelo modelo', f"{encontradas} de {int(teste['real'].sum())}")
    col2.metric('Se escolhesse ao acaso', f'{sorteio:.1f}')
    col3.metric('Ganho sobre o acaso', f'{encontradas / sorteio:.1f}x' if sorteio else '-')
    curva = pd.DataFrame({'Abordados': range(1, len(ordenado) + 1), 'Modelo': ordenado['real'].cumsum().values})
    curva['Ao acaso'] = curva['Abordados'] * teste['real'].mean()
    st.line_chart(curva.set_index('Abordados'), height=320)

with aba_modelo:
    metricas = ficha['metricas_teste']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('PR-AUC (teste)', f"{metricas['PR-AUC']:.2f}", help=f"Um sorteio teria {pacote['taxa_base']:.2f}")
    col2.metric('Recall', f"{metricas['Recall (Sensibilidade)']:.0%}", help='Das pessoas que saíram, quantas o modelo apontou')
    col3.metric('Precisão', f"{metricas['Precisão']:.0%}", help='Das pessoas apontadas, quantas saíram de fato')
    col4.metric('Limiar', f'{limiar:.0%}')
    st.markdown(f"**Modelo:** {ficha['modelo']}  \n**Treinado com:** {ficha['treinado_com']}")
    st.markdown('**Avisos:**\n' + '\n'.join(f'- {aviso}' for aviso in ficha['avisos']))
