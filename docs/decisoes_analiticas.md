# Registro de decisões analíticas

Este documento registra as principais decisões do projeto, no formato
**Contexto → Decisão → Alternativas consideradas → Consequência → Onde está**.
Cada decisão tem um código (DA-XX) para ser citada em commits e pull requests.

Os números vêm das saídas do notebook [`Notebooks/Previsão de rotatividade de funcionários - v2.ipynb`](../Notebooks/).

| Código | Decisão |
|---|---|
| [DA-01](#da-01--um-notebook-só-no-roteiro-crisp-dm) | Um notebook só, no roteiro CRISP-DM |
| [DA-02](#da-02--pr-auc-da-classe-saiu-como-métrica-principal) | PR-AUC da classe "saiu" como métrica principal |
| [DA-03](#da-03--transformações-dentro-de-um-pipeline) | Transformações dentro de um `Pipeline` |
| [DA-04](#da-04--separação-602020-estratificada-e-validação-cruzada-repetida) | Separação 60/20/20 estratificada e validação cruzada repetida |
| [DA-05](#da-05--sem-balanceamento-o-desbalanceamento-vai-para-o-limiar) | Sem balanceamento: o desbalanceamento vai para o limiar |
| [DA-06](#da-06--colunas-de-taxa-fora-do-modelo) | Colunas de "taxa" fora do modelo |
| [DA-07](#da-07--três-variáveis-criadas) | Três variáveis criadas |
| [DA-08](#da-08--gênero-fora-do-modelo) | Gênero fora do modelo |
| [DA-09](#da-09--regressão-logística-escolhida-pela-validação-cruzada) | Regressão Logística escolhida pela validação cruzada |
| [DA-10](#da-10--limiar-pelo-custo-escolhido-na-validação) | Limiar pelo custo, escolhido na validação |
| [DA-11](#da-11--avaliação-honesta-e-teste-anti-vazamento) | Avaliação honesta e teste anti-vazamento |

---

## DA-01 — Um notebook só, no roteiro CRISP-DM

- **Contexto:** a versão original tinha dois notebooks (EDA e Modelo de Churn), que repetiam a leitura e a preparação dos dados.
- **Decisão:** um único notebook v2 (`Previsão de rotatividade de funcionários - v2.ipynb`) com EDA e modelo no roteiro CRISP-DM. Decisão do usuário no plano da v2.
- **Alternativas consideradas:** dois notebooks v2 espelhando a v1.
- **Consequência:** a preparação é feita uma vez, em `src/dados.py`. Os dois notebooks originais ficam intactos.
- **Onde está:** `Notebooks/`.

## DA-02 — PR-AUC da classe "saiu" como métrica principal

- **Contexto:** 16,1% dos funcionários saíram. Quem chuta "todo mundo fica" acerta 83,9% (84,0% no teste). A versão original divulgou 84% de acurácia como resultado do modelo.
- **Decisão:** comparar e ajustar os modelos pela **PR-AUC** da classe "saiu", com recall, precisão, F1 e o custo em dólares como apoio.
- **Alternativas consideradas:** acurácia (não discrimina); F1 (depende do limiar); AUC-ROC (fica alta mesmo com precisão baixa).
- **Consequência:** o chute tem PR-AUC de 0,16; o modelo final, 0,587 no teste (IC 95% de 0,47 a 0,70).
- **Onde está:** `calcular_metricas_completas` em `src/avaliacao.py`; item 2.7 da Fase 02 e itens 5 a 7 da Fase 03.

## DA-03 — Transformações dentro de um `Pipeline`

- **Contexto:** a versão original reajustava o OneHotEncoder e o MinMaxScaler separadamente na validação e no teste (cada conjunto numa escala) e salvou para produção os ajustados **no teste**.
- **Decisão:** one-hot e padronização dentro de um `Pipeline` (`ColumnTransformer`), ajustado só no treino e salvo junto com o modelo.
- **Alternativas consideradas:** nenhuma razoável.
- **Consequência:** treino, validação, teste, validação cruzada e app usam exatamente as mesmas transformações.
- **Onde está:** `preprocessador` (item 3 da Fase 03); `Model/modelo_rotatividade.pkl`.

## DA-04 — Separação 60/20/20 estratificada e validação cruzada repetida

- **Contexto:** a base é pequena (1.470 funcionários) e a validação e o teste têm só 47 saídas cada. Na versão original, a validação não era estratificada.
- **Decisão:** treino, validação e teste estratificados (882 / 294 / 294) e validação cruzada estratificada com 5 dobras × 3 repetições para escolher o balanceamento, as variáveis e o modelo. O teste é usado uma única vez, com intervalo de confiança por bootstrap.
- **Alternativas consideradas:** decidir só pela validação (o LightGBM parecia o melhor, 0,598, mas fica em 0,564 na validação cruzada).
- **Consequência:** as escolhas não dependem da sorte de uma divisão.
- **Onde está:** itens 4.1, 4.2, 4.3 e 5 da Fase 03; item 1.1 da Fase 04.

## DA-05 — Sem balanceamento: o desbalanceamento vai para o limiar

- **Contexto:** o plano previa peso da classe, comparado com SMOTE. Na validação cruzada da Regressão Logística, sem balanceamento: PR-AUC de 0,638; peso da classe e SMOTE, 0,597; ClusterCentroids (v1), 0,590. No XGBoost, o ClusterCentroids cai para 0,385. Com peso da classe, as probabilidades ficaram distorcidas (quem recebia 41% de risco saía 11% das vezes) e o Brier ficou pior que o do chute.
- **Decisão:** treinar sem balanceamento e tratar o desbalanceamento no limiar, escolhido pelo custo (DA-10).
- **Alternativas consideradas:** peso da classe, SMOTE, ClusterCentroids (versão original, que descartava 82% dos funcionários que ficaram).
- **Consequência:** probabilidades calibradas (Brier de 0,099 contra 0,134 do chute), essenciais para a conta de custo.
- **Onde está:** item 4.2 e item 8.4 da Fase 03.

## DA-06 — Colunas de "taxa" fora do modelo

- **Contexto:** `taxa_diaria`, `taxa_hora` e `taxa_mensal` não têm relação com o salário (correlações de 0,02, −0,02 e 0,05) nem com a saída (efeitos de −0,09 a 0,02). A versão original descrevia `DailyRate` como "cálculo do salário". O Random Forest as coloca entre as 10 mais importantes (viés por variáveis com muitos valores); a Informação Mútua, em 17º, 29º e 53º.
- **Decisão:** retirá-las do modelo.
- **Alternativas consideradas:** mantê-las (PR-AUC de 0,620 com elas, 0,624 sem).
- **Consequência:** modelo mais simples, sem perda.
- **Onde está:** `TAXAS` em `src/config.py`; H12 e item 4.3 da Fase 03.

## DA-07 — Três variáveis criadas

- **Contexto:** cada satisfação sozinha tem efeito pequeno; quem saiu tem metade dos anos por empresa de quem ficou (1,62 contra 3,33).
- **Decisão:** criar `satisfacao_media`, `anos_por_empresa` e `fracao_carreira_na_empresa`.
- **Alternativas consideradas:** usar só as originais.
- **Consequência:** a PR-AUC na validação cruzada sobe de 0,624 para 0,643; a satisfação média é a 3ª variável mais importante no teste.
- **Onde está:** `criar_variaveis` em `src/dados.py`; item 4 da Fase 02.

## DA-08 — Gênero fora do modelo

- **Contexto:** na base, homens e mulheres saem na mesma proporção (17,0% e 14,8%; V de Cramér de 0,01). Mesmo assim, com o gênero, o modelo dava às mulheres uma chance de sair 24% menor que a dos homens com todo o resto igual (0,76 vez). Usar gênero para decidir com quem o RH conversa é um problema ético e legal (discriminação, LGPD).
- **Decisão:** retirar o gênero do modelo, mantendo-o na EDA (H13) e na conferência por grupo da Fase 04. Decisão aprovada pelo usuário.
- **Alternativas consideradas:** manter e registrar o risco na ficha do modelo.
- **Consequência:** na auditoria da Fase 04, com o mesmo processo, a PR-AUC na validação cruzada vai de 0,650 (com gênero) para 0,647 (sem) e no teste de 0,598 para 0,587, diferenças menores que a incerteza. Com o limiar de menor custo, o modelo sem gênero encontra 23 das 47 saídas do teste (21 com gênero) e economiza 32,8% (30,2% com gênero). O recall é de 56% nas mulheres e 45% nos homens, dentro da incerteza (16 e 31 saídas). O app e a ficha avisam para não reintroduzir o gênero.
- **Onde está:** `SENSIVEIS` em `src/config.py`; item 4.3 da Fase 03; item 2 da Fase 04.

## DA-09 — Regressão Logística escolhida pela validação cruzada

- **Contexto:** na validação cruzada, a Regressão Logística tem PR-AUC de 0,648 ± 0,046, contra 0,564 do LightGBM, 0,561 do XGBoost, 0,542 do Random Forest e 0,410 do Naive Bayes. As árvores decoram o treino (PR-AUC de 1,00).
- **Decisão:** Regressão Logística ajustada (C = 0,18, penalidade L2), retreinada em treino + validação (`clone`).
- **Alternativas consideradas:** LightGBM (melhor só na validação isolada); Random Forest (escolha "100%" da versão original).
- **Consequência:** modelo estável e interpretável: cada coeficiente vira um multiplicador da chance (hora extra: 2,09x).
- **Onde está:** itens 5 a 7 da Fase 03; item 3 da Fase 04.

## DA-10 — Limiar pelo custo, escolhido na validação

- **Contexto:** com o limiar padrão de 0,50, o modelo encontra só 32% das saídas no teste. A versão original usava 0,50 e uma conta de negócio sem base ("214 funcionários = US$ 1.636.030").
- **Decisão:** limiar de 0,34, o de menor custo na validação, na premissa de que repor custa US$ 7.645 e abordar custa 1/5 disso (premissa escolhida pelo usuário). A sensibilidade a outras proporções fica registrada.
- **Alternativas consideradas:** F1 máximo (0,41; custo parecido); meta de 70% de recall (0,18; aborda o dobro de pessoas).
- **Consequência:** no teste, o RH aborda 38 pessoas e encontra 23 das 47 saídas; o custo cai 32,8% (US$ 117.733) em relação a não ter modelo.
- **Onde está:** `curva_custo` e `resumo_custo` em `src/avaliacao.py`; `CUSTO_REPOSICAO` e `PROPORCAO_CUSTO` em `src/config.py`; item 8 da Fase 03.

## DA-11 — Avaliação honesta e teste anti-vazamento

- **Contexto:** a versão original treinou os modelos "ajustados" em treino + validação e mediu na própria validação: o Random Forest tirou 100% em tudo.
- **Decisão:** reproduzir o problema e mostrar o resultado honesto ao lado; o teste é aberto uma única vez.
- **Alternativas consideradas:** nenhuma razoável.
- **Consequência:** o mesmo Random Forest tem PR-AUC de 0,555 na avaliação honesta, contra 1,000 com vazamento.
- **Onde está:** item 9 da Fase 03.
