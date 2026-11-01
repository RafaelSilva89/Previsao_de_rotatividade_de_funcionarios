# 👥 Previsão de Rotatividade de Funcionários

**Quem tem mais chance de sair da empresa, por quê, e com quem o RH deve conversar primeiro.**

<div align="center">
<img src="Imagens/Capa_desempenho_funcionario.PNG" width="450px">
</div>

[![Testes](https://github.com/RafaelSilva89/Previsao_de_rotatividade_de_funcionarios/actions/workflows/testes.yml/badge.svg)](https://github.com/RafaelSilva89/Previsao_de_rotatividade_de_funcionarios/actions/workflows/testes.yml)

**Python · Pandas · Scikit-learn · Regressão Logística · XGBoost · LightGBM · Optuna · SHAP · SciPy · Pytest · GitHub Actions · Streamlit · CRISP-DM**

| 📊 Destaques | |
|---|---|
| Base | **1.470 funcionários** (IBM HR Analytics, dados fictícios), dos quais **16,1% saíram** |
| Maior fator de risco | **Hora extra**: chance de sair **3,8x** maior; no cargo júnior (nível 1) com hora extra, **52,6%** saem |
| Modelo | Abordando as **30 pessoas** de maior risco (10% do teste), o RH encontra **20 das 47 saídas**: **4x mais** que ao acaso |
| Resultado em dólares | Com o limiar escolhido pelo custo, a empresa gasta **32,8% menos** com reposição no teste (US$ 117.733), numa premissa de custo explícita |
| Qualidade | A v2 mostrou que a **acurácia de 84% da v1 era o acerto de quem chuta "todo mundo fica"**, que o "Random Forest com 100%" foi avaliado em dados já vistos, e **tirou o gênero do modelo** sem perda de desempenho |
| Ajuste e explicação | **Grid Search, Random Search e otimização Bayesiana (Optuna)** comparados nas mesmas 5 dobras; **SHAP** explica por que cada pessoa foi apontada; **curvas ROC e Precision-Recall**; **monitoramento de drift** (PSI e K-S) com SLA |
| Entrega | Notebook CRISP-DM, **19 testes automatizados** (GitHub Actions) e **app Streamlit** com o risco de cada funcionário e a lista de prioridade do RH |

---

## 📑 Sumário

1. [Contexto do problema](#1-contexto-do-problema)
2. [Objetivo](#2-objetivo)
3. [Base de dados](#3-base-de-dados)
4. [Estrutura do projeto](#4-estrutura-do-projeto)
5. [Como executar](#5-como-executar)
6. [Etapas do projeto (CRISP-DM)](#6-etapas-do-projeto-crisp-dm)
7. [Principais descobertas da análise exploratória](#7-principais-descobertas-da-análise-exploratória)
8. [Construção do modelo](#8-construção-do-modelo)
9. [Resultado do modelo](#9-resultado-do-modelo)
10. [App interativo (Streamlit)](#10-app-interativo-streamlit)
11. [O que mudou da v1 para a v2](#11-o-que-mudou-da-v1-para-a-v2)
12. [Insights](#12-insights)
13. [Limites](#13-limites)
14. [Próximos passos](#14-próximos-passos)

---

## 1. Contexto do problema

<img src="Imagens/Capa2_desempenho_funcionario.PNG" width="180px" align="right">

Perder um funcionário custa caro: recrutamento, tempo até a vaga ser ocupada e treinamento do substituto. Uma estimativa citada na versão original é de **US$ 7.645** por contratação, com cerca de 52 dias até a nova pessoa assumir ([fonte](https://toggl.com/blog/cost-of-hiring-an-employee)).

O RH não consegue conversar com todos ao mesmo tempo. Ele precisa saber **com quem falar primeiro** e **por quê**.

## 2. Objetivo

> **Perguntas de negócio:**
> 1. **O que está associado à saída** de um funcionário?
> 2. Dá para **prever quem vai sair**? Com que precisão, comparado a escolher ao acaso?
> 3. Com quantas pessoas o RH deve **conversar** para que a ação compense o custo?
> 4. Quanto a empresa **economiza** usando o modelo?

## 3. Base de dados

A base é a [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle), salva em `Datasets/Human_Resources.csv`. Ela é **fictícia**: foi criada por cientistas de dados da IBM para ensino.

| Grupo | Colunas (traduzidas) |
|---|---|
| 🎯 Alvo | `saiu` (1 = saiu da empresa) |
| Perfil | idade, gênero, estado civil, escolaridade, área de formação, distância de casa |
| Cargo e salário | departamento, cargo, nível do cargo (1 a 5), salário mensal, último aumento, nível de ações |
| Rotina | hora extra, viagens a trabalho |
| Pesquisas internas | satisfação com o ambiente, o trabalho e o relacionamento, equilíbrio vida-trabalho, envolvimento (1 a 4) |
| Carreira | anos de experiência, de empresa, no cargo, com o gestor, desde a última promoção, empresas anteriores |
| ⚠️ Sem uso | 3 colunas constantes, o número de registro e 3 "taxas" sem relação com o salário |

## 4. Estrutura do projeto

```
Previsao_de_rotatividade_de_funcionarios/
├── Datasets/
│   └── Human_Resources.csv             # base original (IBM HR Analytics)
├── Imagens/                            # gráficos gerados pelos notebooks
├── Model/
│   ├── modelo_rotatividade.pkl         # pipeline final + limiar + valores típicos (usado pelo app)
│   ├── modelo_rotatividade.json        # ficha do modelo (métricas, premissa de custo, avisos)
│   └── variaveis_modelo.pkl            # artefatos da versão original
├── docs/
│   └── decisoes_analiticas.md          # registro das decisões do projeto (DA-01 a DA-11)
├── reports/
│   ├── comparacao_modelos.csv          # métricas dos 5 algoritmos na validação
│   └── previsoes_teste.csv             # probabilidades do teste (usadas pelo app)
├── Notebooks/
│   ├── Previsão de rotatividade de funcionários - EDA.ipynb             # versão original (análise)
│   ├── Previsão de rotatividade de funcionários - Modelo de Churn.ipynb # versão original (modelo)
│   └── Previsão de rotatividade de funcionários - v2.ipynb              # v2: análise + modelo (CRISP-DM)
├── src/
│   ├── config.py                       # caminhos, semente, colunas e premissa de custo
│   ├── dados.py                        # leitura, tradução, variáveis criadas e preparo de um funcionário
│   ├── estatistica.py                  # testes das hipóteses (com tamanho do efeito) e regra do IQR
│   ├── selecao.py                      # seleção de variáveis (Random Forest + Informação Mútua)
│   ├── avaliacao.py                    # métricas, limiar, custo, ganho sobre o acaso, bootstrap, ficha e explicação de cada previsão
│   ├── otimizacao.py                   # comparação Grid Search x Random Search x Bayesiana (Optuna)
│   ├── monitoramento.py                # data drift: PSI e teste K-S
│   └── graficos.py                     # gráficos e formatação em português
├── tests/
│   ├── test_funcoes.py                 # 15 testes das funções de src/
│   └── test_app.py                     # 4 testes do app (AppTest do Streamlit)
├── .github/workflows/testes.yml        # roda os testes a cada push (GitHub Actions)
├── app.py                              # app Streamlit
├── pytest.ini
├── requirements.txt
└── README.md
```

## 5. Como executar

```bash
# 1. Clonar o repositório
git clone https://github.com/RafaelSilva89/Previsao_de_rotatividade_de_funcionarios.git
cd Previsao_de_rotatividade_de_funcionarios

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Rodar os testes
pytest

# 4. Abrir o notebook e executar todas as células (gera o modelo usado pelo app)
jupyter notebook "Notebooks/Previsão de rotatividade de funcionários - v2.ipynb"

# 5. Abrir o app
streamlit run app.py
```

### ⏱️ Tempo de execução do notebook

A execução completa leva **cerca de 6 minutos**. Tempo de cada etapa:

| Etapa | Tempo |
|---|---|
| Fases 01 e 02: entendimento, limpeza e hipóteses | 0,6 min |
| Fase 03: balanceamento (comparação das técnicas) | 1,3 min |
| Fase 03: comparação dos modelos (validação cruzada) | ~1 min |
| Fase 03: 7.1 Grid Search | 0,4 min |
| Fase 03: 7.2 Random Search | < 0,3 min |
| Fase 03: 7.3 Otimização Bayesiana (Optuna) | 1,1 min |
| Fase 03: comparação dos métodos, modelo final e limiar | < 1 min |
| Fase 04: avaliação, curvas ROC/PR, permutação e SHAP | 0,8 min |
| Fase 05: modelo de produção, simulação, monitoramento e app | < 0,1 min |
| **Total** | **5,6 min** |

A base é pequena (1.470 funcionários) e o modelo é linear: o notebook inteiro roda rápido.

Medido na execução completa de 30/10/2026, num notebook com Intel Core i3-4005U (2 núcleos, 1,7 GHz) e 8 GB de RAM, com o kernel em Python 3.13. Num computador mais novo o tempo cai bastante, mas a proporção entre as etapas se mantém.

## 6. Etapas do projeto (CRISP-DM)

| Fase | O que foi feito |
|---|---|
| **01. Entendimento do negócio** | Definição das 4 perguntas e da premissa de custo |
| **02. Preparação dos dados** | Tradução da base, colunas sem informação, inconsistências, outliers pela regra do IQR e EDA com 13 hipóteses (cada uma com teste estatístico e tamanho do efeito) |
| **03. Modelagem** | Divisão 60/20/20 estratificada, 4 estratégias de balanceamento, seleção de variáveis, 5 algoritmos com validação cruzada, ajuste fino com Grid, Random e Bayesiana, limiar pelo custo, calibração e teste anti-vazamento |
| **04. Avaliação** | Resultado no teste com intervalo de confiança, em dólares e por grupo (departamento, hora extra, gênero), curvas ROC e Precision-Recall, e variáveis mais importantes (permutação e SHAP) |
| **05. Implantação** | Modelo salvo com ficha, simulação de perfis, curva "modelo × sorteio", monitoramento de drift com SLA e app Streamlit |

## 7. Principais descobertas da análise exploratória

<div align="center">
<img src="Imagens/H8 Nivel x hora extra.png" width="600px">
</div>

| # | Hipótese | Resultado | Evidência |
|---|---|---|---|
| **H1** | Hora extra → sai mais | ✅ Verdadeira | Chance **3,8x** maior |
| **H2** | Mais jovens saem mais | ✅ Verdadeira | Idade mediana de **32** contra **36** anos |
| **H3** | Quem ganha menos sai mais | ✅ Verdadeira | **US$ 3.202** contra **US$ 5.204** |
| **H4** | Solteiros saem mais | ✅ Verdadeira | Efeito pequeno |
| **H5** | Quem viaja muito sai mais | ✅ Efeito pequeno | — |
| **H6** | Quem mora longe sai mais | ⚠️ Efeito pequeno | — |
| **H7** | Representantes de vendas saem mais | ✅ Verdadeira | **39,8%** saem, contra **2,5%** dos diretores de pesquisa |
| **H8** | Júnior + hora extra multiplicam o risco | ✅ **Maior efeito** | **52,6%** saem no nível 1 com hora extra |
| **H9** | Menos satisfeito → sai mais | ⚠️ Parcial | A satisfação **média** pesa mais que cada pesquisa |
| **H10** | Pouco tempo de casa → sai mais | ✅ Verdadeira | Mas anos sem promoção, sozinho, quase não difere |
| **H11** | Sem ações → sai mais | ✅ Confundida | **100%** dos solteiros estão sem ações |
| **H12** | As "taxas" são ruído | ✅ Verdadeira | Correlação com o salário de ~**0** |
| **H13** | Homens e mulheres saem igual | ✅ Verdadeira | **14,8%** contra **17,0%** (p = 0,29) |

> 🧪 **Testes estatísticos:** qui-quadrado + V de Cramér (e odds ratio) para as categorias, Mann-Whitney + correlação de postos para as numéricas. Com efeitos de 0,1 a 0,36, quase todos são **pequenos a médios**: nenhuma variável sozinha explica a saída. O que importa é o tamanho do efeito, não o p.

## 8. Construção do modelo

| Etapa | Como foi feito |
|---|---|
| **Pipeline** | One-hot das categorias e padronização das numéricas, ajustados só no treino e salvos junto com o modelo |
| **Divisão** | 60% treino (882), 20% validação (294) e 20% teste (294), estratificada; o teste é aberto uma única vez |
| **Balanceamento** | Comparamos 4 estratégias. **Sem balanceamento** venceu (PR-AUC de 0,638 contra 0,597 com peso da classe) e manteve as probabilidades realistas. O ClusterCentroids da v1, que descartava 82% dos dados, foi o pior |
| **Variáveis** | 29: sem as 3 "taxas" (ruído), com 3 criadas (satisfação média, anos por empresa, fração da carreira na empresa) e **sem o gênero** (atributo sensível; veja abaixo) |
| **Algoritmos** | Regressão Logística, Naive Bayes, Random Forest, XGBoost e LightGBM, com validação cruzada de 5 dobras × 3 repetições |
| **Ajuste fino** | Grid Search, Random Search e Bayesiana (Optuna) comparados (tabela abaixo): **empate técnico**; o Random Search ficou (C = 0,18, L2) |
| **Escolha** | **Regressão Logística** ajustada: PR-AUC de **0,648 ± 0,046**, contra 0,564 do segundo colocado. As árvores decoravam o treino |
| **Limiar** | **0,34**, o de menor custo na validação: repor custa US$ 7.645 e abordar custa 1/5 disso |
| **Anti-vazamento** | Avaliado em dados já vistos, o Random Forest tira **100%** em tudo; de forma honesta, PR-AUC de **0,555** |

<div align="center">
<img src="Imagens/Curva de custo.png" width="700px">
</div>

### Ajuste fino: Grid Search x Random Search x Bayesiana

| Método | Como escolhe | PR-AUC (validação cruzada) | Custo |
|---|---|---|---|
| Sem ajuste (C = 1) | — | 0,654 ± 0,060 | — |
| **Grid Search** | 11 valores de C x L1/L2 (todas as 22 combinações) | 0,664 ± 0,049 | 110 modelos, 19 s |
| **Random Search** | 30 combinações sorteadas, C contínuo | **0,664 ± 0,054** | 150 modelos, **11 s** |
| **Bayesiana (Optuna)** | 30 tentativas guiadas, com a mistura L1/L2 (*elastic net*) | 0,661 ± 0,047 | 150 modelos, 65 s |

Os três chegam à mesma resposta (regularização L2 mais forte que a padrão) e ficam a 0,003 um do outro, 1/20 do desvio entre as dobras. Com só 2 hiperparâmetros, o método mais simples basta; a Bayesiana compensa em espaços grandes e modelos caros. O modelo e o limiar não mudaram.

<div align="center">
<img src="Imagens/Grid search.png" width="650px">
</div>

### ⚖️ Por que o gênero ficou fora do modelo (e quanto isso custou)

Na base, homens e mulheres saem na mesma proporção (17,0% e 14,8%; H13). Mesmo assim, quando recebia o gênero, o modelo dava às mulheres uma chance de sair **24% menor** que a dos homens com todo o resto igual. Usar gênero para decidir com quem o RH conversa é um problema ético e legal (discriminação, LGPD), então ele foi retirado. A auditoria abaixo treina o mesmo modelo com e sem o gênero, pelo mesmo processo, e compara no teste:

| | Com gênero | Sem gênero (modelo final) |
|---|---|---|
| PR-AUC (validação cruzada do treino) | 0,650 | **0,647** |
| PR-AUC (teste) | 0,598 | **0,587** |
| Saídas encontradas no teste (limiar de menor custo) | 21 de 47 | **23 de 47** |
| F1 no teste | 0,519 | **0,541** |
| Economia vs. sem modelo | 30,2% | **32,8%** |

**Resultado:** a qualidade da ordem de risco cai muito pouco (0,003 a 0,011 de PR-AUC, bem menos que a incerteza do teste), e o resultado prático, com o limiar pelo custo, até melhora um pouco. As diferenças são ruído de uma base pequena: **retirar o gênero não teve custo real de desempenho**.

## 9. Resultado do modelo

| Teste (294 funcionários, 47 saídas) | Modelo (limiar 0,34) | Chutar "todo mundo fica" |
|---|---|---|
| Saídas encontradas (recall) | **23 de 47 (49%)** | 0 |
| Dos apontados, quantos saem (precisão) | **61%** | — |
| PR-AUC | **0,587** (IC 95%: 0,47 a 0,70) | 0,16 |
| Acurácia | 86,7% | 84,0% |
| Custo com reposição | **US$ 241.582** | US$ 359.315 |

- **4x mais eficiente que o acaso:** abordando 30 pessoas, o RH encontra 20 saídas, contra 4,8 num sorteio.
- **Bem calibrado:** quem recebe 10% a 20% de risco sai 13,8% das vezes.
- **Onde erra:** encontra 88% das saídas em Vendas, mas só 31% em Pesquisa e Desenvolvimento, e não prevê saídas de veteranos com décadas de casa (provavelmente aposentadorias).
- **Justiça:** recall de 56% nas mulheres e 45% nos homens, diferença dentro da incerteza do teste.

<div align="center">
<img src="Imagens/Odds ratio regressao logistica.png" width="650px">
</div>

**Por que cada pessoa foi apontada (SHAP):** o SHAP parte do risco médio e soma a contribuição de cada variável até o risco daquela pessoa. A **hora extra** domina (com o resto igual, a chance de quem faz é ~5 vezes a de quem não faz), seguida de **anos sem promoção** e **anos por empresa**. Exemplo: o representante de vendas de 21 anos com 97% de risco soma hora extra (+1,16), idade (+0,58) e insatisfação com o ambiente (+0,40), além de muitos sinais pequenos. As 24 saídas não previstas quase não se movem do risco médio: o modelo não tem sinal para elas. Mesmo sem gênero, o risco médio previsto ficou em 17,0% para mulheres e 14,7% para homens (taxas reais: 13,8% e 17,4%), dentro da incerteza, mas registrado para acompanhar.

<div align="center">
<img src="Imagens/SHAP waterfall saiu.png" width="650px">
</div>

**Monitoramento:** as entradas novas são comparadas com as do treino (**PSI** e **K-S**) e a PR-AUC tem um **SLA** de 0,47. Numa simulação em que a hora extra passa de 26% para 63% dos funcionários, o PSI marca mudança forte (0,50) e o número de apontados quase dobra (38 → 71).

## 10. App interativo (Streamlit)

O app tem 3 abas:
1. **🧑 Funcionário:** informe o perfil (idade, salário, hora extra, cargo, satisfação...) e receba a **probabilidade de saída**, a comparação com a taxa média e a recomendação (conversar ou acompanhar). Logo abaixo, o app mostra **as 3 variáveis que mais pesaram** para aquela pessoa, comparada ao funcionário típico da empresa. Exemplo: *"Hora extra: sim (típico: não) → multiplica a chance de sair por 5,1"*. Assim o RH sabe **o que conversar**. É o que o modelo associa ao risco, não necessariamente a causa.
2. **📋 Lista de prioridade do RH:** escolha quantas pessoas o RH consegue abordar e veja **quantas saídas** o modelo encontraria no teste, contra o acaso.
3. **ℹ️ Sobre o modelo:** métricas, limiar e avisos de uso.

```bash
streamlit run app.py
```

> ℹ️ O app usa `Model/modelo_rotatividade.pkl` e `reports/previsoes_teste.csv`, gerados ao executar o notebook até a Fase 05.

## 11. O que mudou da v1 para a v2

Os dois notebooks originais foram mantidos em `Notebooks/` para mostrar a evolução.

| # | Versão original (v1) | Problema | Versão v2 |
|---|---|---|---|
| 1 | "84% de chance de acertar" | É o acerto de quem chuta "todo mundo fica" (84,0% no teste) | Métrica principal: **PR-AUC** da classe "saiu", com recall, precisão e custo |
| 2 | Encoder e scaler reajustados na validação e no teste; `.pkl` salvo com os ajustados no teste | Cada conjunto numa escala; o app aplicaria outra transformação | Tudo num `Pipeline` ajustado só no treino |
| 3 | Random Forest com **100%** em tudo | Treinado em treino + validação e avaliado na validação | Reproduzido no teste anti-vazamento; honesto: PR-AUC de 0,555 |
| 4 | ClusterCentroids no treino | Descartava **82%** dos funcionários que ficaram | 4 estratégias comparadas; **sem balanceamento** venceu, com o limiar tratando o desbalanceamento |
| 5 | GridSearch pela acurácia, 2.800 combinações, 30 minutos | Métrica errada; parâmetros inválidos (`bootstrap='False'` em texto) | Grid, Random e Bayesiana comparados pela PR-AUC; Random Search com 30 combinações em 11 segundos |
| 6 | Limiar fixo em 0,50 | Encontra só 32% das saídas | Limiar de **0,34** pelo custo, escolhido na validação |
| 7 | "214 funcionários = US$ 1.636.030" | Conta sem base (acurácia × 500) | Economia de **32,8%** no teste, com a premissa explícita e a sensibilidade |
| 8 | Taxas tratadas como salário | Não têm relação com o salário | Fora do modelo (H12) |
| 9 | Gênero no modelo | Atributo sensível | Fora do modelo; justiça conferida por grupo |
| 10 | Dois notebooks, hipóteses só com gráficos | Preparação repetida; efeito real não diferenciado do acaso | Um notebook CRISP-DM com **13 hipóteses** testadas e tamanho do efeito |
| 11 | Sem testes nem registro das decisões | — | `src/`, **19 testes** no GitHub Actions, `docs/decisoes_analiticas.md`, ficha do modelo e app |

## 12. Insights

1. **Comparar sempre com o chute:** a acurácia de 84% da v1 era o acerto de quem diz que ninguém sai.
2. **Júnior com hora extra é o grupo de maior risco** (52,6% saem): rever a carga de trabalho é a ação mais direta.
3. **Ficar muito tempo sem promoção pesa, com o resto igual** (1,63x a chance), embora sozinho pareça irrelevante.
4. **Representantes de vendas saem 39,8% das vezes**, mais que o dobro da média.
5. **Balancear piorou:** o peso da classe distorceu as probabilidades, e o ClusterCentroids jogou fora dados.
6. **Com poucos dados, o modelo simples venceu:** a Regressão Logística superou as árvores e é a mais fácil de explicar.
7. **Abordando 10% dos funcionários, o RH encontra 43% das saídas.**
8. **Atributos sensíveis fora:** sem o gênero, o modelo perdeu nada relevante.
9. **Mais ajuste não é melhor modelo:** Grid, Random e Bayesiana empatam; com poucos hiperparâmetros, a escolha do método de busca importa pouco.

## 13. Limites

- **Base fictícia** (IBM): as conclusões valem para esta base; numa empresa real, o modelo precisa ser retreinado.
- **Base pequena:** com 47 saídas no teste, o recall fica entre 34% e 64% (IC 95%).
- **Fotografia, não filme:** sem datas, o modelo aponta associação, não causa.
- **Aposentadorias** não são previstas.
- **Premissa de custo:** a economia supõe que a conversa evita a saída.
- **Uso ético:** priorizar conversas de retenção, nunca punir ou desligar.

## 14. Próximos passos

1. Retreinar com **dados reais e datas**, separando pedidos de demissão de aposentadorias e prevendo **quando** a saída acontece.
2. **Medir o efeito da conversa** de retenção com um teste controlado.
3. Acompanhar a **justiça do modelo** por gênero, idade e cargo a cada retreino.
4. Publicar o app no Streamlit Community Cloud.
