"""Constantes do projeto: caminhos, semente, colunas e premissas de negócio."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO_BASE = RAIZ / 'Datasets' / 'Human_Resources.csv'
PASTA_MODELO = RAIZ / 'Model'
PASTA_REPORTS = RAIZ / 'reports'

SEMENTE = 42
ALVO = 'saiu'

# Tradução das colunas da IBM para português
COLUNAS = {
    'Age': 'idade', 'Attrition': 'saiu', 'BusinessTravel': 'viagens', 'DailyRate': 'taxa_diaria',
    'Department': 'departamento', 'DistanceFromHome': 'distancia_casa', 'Education': 'escolaridade',
    'EducationField': 'area_formacao', 'EmployeeCount': 'contagem', 'EmployeeNumber': 'id_funcionario',
    'EnvironmentSatisfaction': 'satisfacao_ambiente', 'Gender': 'genero', 'HourlyRate': 'taxa_hora',
    'JobInvolvement': 'envolvimento', 'JobLevel': 'nivel_cargo', 'JobRole': 'cargo',
    'JobSatisfaction': 'satisfacao_trabalho', 'MaritalStatus': 'estado_civil', 'MonthlyIncome': 'salario_mensal',
    'MonthlyRate': 'taxa_mensal', 'NumCompaniesWorked': 'empresas_anteriores', 'Over18': 'maior_18',
    'OverTime': 'hora_extra', 'PercentSalaryHike': 'aumento_salarial_pct', 'PerformanceRating': 'avaliacao_desempenho',
    'RelationshipSatisfaction': 'satisfacao_relacionamento', 'StandardHours': 'horas_padrao',
    'StockOptionLevel': 'nivel_acoes', 'TotalWorkingYears': 'anos_experiencia',
    'TrainingTimesLastYear': 'treinamentos_ano', 'WorkLifeBalance': 'equilibrio_vida',
    'YearsAtCompany': 'anos_empresa', 'YearsInCurrentRole': 'anos_cargo_atual',
    'YearsSinceLastPromotion': 'anos_ultima_promocao', 'YearsWithCurrManager': 'anos_gestor_atual',
}

# Tradução dos valores das colunas de texto
VALORES = {
    'viagens': {'Non-Travel': 'Não viaja', 'Travel_Rarely': 'Raramente', 'Travel_Frequently': 'Frequentemente'},
    'departamento': {'Sales': 'Vendas', 'Research & Development': 'Pesquisa e Desenvolvimento',
                     'Human Resources': 'Recursos Humanos'},
    'area_formacao': {'Life Sciences': 'Ciências Biológicas', 'Medical': 'Medicina', 'Marketing': 'Marketing',
                      'Technical Degree': 'Técnico', 'Human Resources': 'Recursos Humanos', 'Other': 'Outra'},
    'genero': {'Male': 'Masculino', 'Female': 'Feminino'},
    'cargo': {'Sales Executive': 'Executivo de Vendas', 'Research Scientist': 'Cientista Pesquisador',
              'Laboratory Technician': 'Técnico de Laboratório', 'Manufacturing Director': 'Diretor de Produção',
              'Healthcare Representative': 'Representante de Saúde', 'Manager': 'Gerente',
              'Sales Representative': 'Representante de Vendas', 'Research Director': 'Diretor de Pesquisa',
              'Human Resources': 'Recursos Humanos'},
    'estado_civil': {'Single': 'Solteiro(a)', 'Married': 'Casado(a)', 'Divorced': 'Divorciado(a)'},
}

# Colunas sem informação: constantes (todos iguais) e o número de registro
SEM_INFORMACAO = ['contagem', 'maior_18', 'horas_padrao', 'id_funcionario']

CATEGORICAS = ['viagens', 'departamento', 'area_formacao', 'genero', 'cargo', 'estado_civil']
TAXAS = ['taxa_diaria', 'taxa_hora', 'taxa_mensal']   # sem relação com o salário: ruído da base sintética
NUMERICAS = ['idade', 'distancia_casa', 'escolaridade', 'satisfacao_ambiente', 'envolvimento', 'nivel_cargo',
             'satisfacao_trabalho', 'salario_mensal', 'empresas_anteriores', 'hora_extra', 'aumento_salarial_pct',
             'avaliacao_desempenho', 'satisfacao_relacionamento', 'nivel_acoes', 'anos_experiencia',
             'treinamentos_ano', 'equilibrio_vida', 'anos_empresa', 'anos_cargo_atual', 'anos_ultima_promocao',
             'anos_gestor_atual']
FEATURES_CRIADAS = ['satisfacao_media', 'anos_por_empresa', 'fracao_carreira_na_empresa']
FEATURES_ORIGINAIS = CATEGORICAS + NUMERICAS + TAXAS
# Atributo sensível: analisado na EDA e na conferência por grupo, mas fora do modelo (decisão ética, DA-08)
SENSIVEIS = ['genero']
# Modelo final: sem as taxas (item 4.3 da Fase 03) e sem o gênero
FEATURES = [coluna for coluna in CATEGORICAS if coluna not in SENSIVEIS] + NUMERICAS + FEATURES_CRIADAS

# Premissa de negócio (fonte da versão original: toggl.com/blog/cost-of-hiring-an-employee):
# repor um funcionário custa US$ 7.645; tentar retê-lo (conversa, plano de carreira, ajuste) custa 1/5 disso.
CUSTO_REPOSICAO = 7_645.0
PROPORCAO_CUSTO = 5
CUSTO_ABORDAGEM = CUSTO_REPOSICAO / PROPORCAO_CUSTO

N_JOBS = 2

# Nomes legíveis das variáveis do modelo (explicação de cada previsão no app)
NOMES_LEGIVEIS = {
    'viagens': 'Viagens a trabalho', 'departamento': 'Departamento', 'area_formacao': 'Área de formação',
    'cargo': 'Cargo', 'estado_civil': 'Estado civil', 'idade': 'Idade', 'distancia_casa': 'Distância de casa',
    'escolaridade': 'Escolaridade', 'satisfacao_ambiente': 'Satisfação com o ambiente', 'envolvimento': 'Envolvimento',
    'nivel_cargo': 'Nível do cargo', 'satisfacao_trabalho': 'Satisfação com o trabalho', 'salario_mensal': 'Salário mensal',
    'empresas_anteriores': 'Empresas anteriores', 'hora_extra': 'Hora extra', 'aumento_salarial_pct': 'Aumento salarial (%)',
    'avaliacao_desempenho': 'Avaliação de desempenho', 'satisfacao_relacionamento': 'Satisfação com os relacionamentos',
    'nivel_acoes': 'Nível de ações da empresa', 'anos_experiencia': 'Anos de experiência', 'treinamentos_ano': 'Treinamentos no ano',
    'equilibrio_vida': 'Equilíbrio vida-trabalho', 'anos_empresa': 'Anos na empresa', 'anos_cargo_atual': 'Anos no cargo atual',
    'anos_ultima_promocao': 'Anos desde a última promoção', 'anos_gestor_atual': 'Anos com o gestor atual',
    'satisfacao_media': 'Satisfação média', 'anos_por_empresa': 'Anos por empresa',
    'fracao_carreira_na_empresa': 'Fração da carreira na empresa',
}
