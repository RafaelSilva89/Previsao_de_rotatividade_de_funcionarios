"""Funções de gráficos e de formatação usadas no notebook. Cada gráfico é salvo na pasta Imagens/ do projeto."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PASTA_IMAGENS = Path(__file__).resolve().parents[1] / 'Imagens'

COR_PRINCIPAL = '#1F4E79'
COR_SAIU = '#C0392B'
COR_FICOU = '#5B8DB8'
COR_NEUTRA = '#B0B7C3'


def salvar(nome_arquivo):
    plt.savefig(PASTA_IMAGENS / f'{nome_arquivo}.png', bbox_inches='tight', dpi=110)


def titulo(texto, subtitulo=None):
    """Título e subtítulo alinhados à esquerda, acima da área do gráfico (espaçamento em polegadas)."""
    altura = plt.gcf().get_figheight()
    topo = plt.gcf().subplotpars.top
    texto = texto.replace('$', r'\$')   # "US$" não pode virar fórmula matemática do matplotlib
    subtitulo = subtitulo.replace('$', r'\$') if subtitulo else subtitulo
    plt.figtext(0.01, topo + 0.45 / altura, texto, ha='left', va='bottom', fontsize=16)
    if subtitulo:
        plt.figtext(0.01, topo + 0.15 / altura, subtitulo, ha='left', va='bottom', fontsize=11, color='#656565',
                    style='italic')


def pct(valor, casas=0):
    """0.1234 -> '12%' (ou '12,3%' com casas=1)."""
    return f'{valor:.{casas}%}'.replace('.', ',')


def dolares(valor, casas=0):
    """7645.5 -> 'US$ 7.646' (separador de milhar com ponto)."""
    texto = f'{valor:,.{casas}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    return f'US$ {texto}'


def plot_barras_h(serie, titulo_grafico, nome_arquivo, subtitulo=None, formato=pct, destaque=None, referencia=None):
    """Barras horizontais ordenadas, com o valor escrito em cada barra.

    destaque: rótulos pintados com a cor de destaque. referencia: valor desenhado como linha tracejada.
    """
    serie = serie.sort_values()
    cores = [COR_SAIU if destaque and rotulo in destaque else COR_PRINCIPAL for rotulo in serie.index]
    plt.figure(figsize=(10, max(4, 0.4 * len(serie))))
    barras = plt.barh(serie.index.astype(str), serie.values, color=cores)
    plt.bar_label(barras, labels=[formato(v) for v in serie.values], padding=3, fontsize=9)
    if referencia is not None:
        plt.axvline(referencia, color='black', ls='--', lw=1)
        plt.annotate(f'geral: {formato(referencia)}', (referencia, len(serie) - 0.4), xytext=(4, 0),
                     textcoords='offset points', fontsize=9)
    plt.xticks([])
    sns.despine(left=True, bottom=True)
    titulo(titulo_grafico, subtitulo)
    salvar(nome_arquivo)
    plt.show()


def plot_taxa_saida(df, coluna, titulo_grafico, nome_arquivo, alvo='saiu', subtitulo=None, ordem=None, ax=None,
                    legenda=True):
    """Taxa de saída por grupo (barras), com a quantidade de pessoas no rótulo e a taxa geral tracejada."""
    tabela = df.groupby(coluna, observed=True)[alvo].agg(['mean', 'size'])
    if ordem is not None:
        tabela = tabela.reindex(ordem)
    sozinho = ax is None
    if sozinho:
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(tabela)), 4.5))
    barras = ax.bar(tabela.index.astype(str), tabela['mean'], color=COR_SAIU, alpha=0.85)
    rotulo = '{}\n({} pessoas)' if sozinho else '{}\n({})'   # nos painéis pequenos, só o número
    ax.bar_label(barras, labels=[rotulo.format(pct(m), n) for m, n in zip(tabela['mean'], tabela['size'])],
                 padding=2, fontsize=9)
    ax.axhline(df[alvo].mean(), color='black', ls='--', lw=1, label=f'taxa geral: {pct(df[alvo].mean(), 1)}')
    ax.set_ylim(0, tabela['mean'].max() * 1.35)
    ax.set_yticks([])
    ax.set_xlabel('')
    if legenda:
        ax.legend(frameon=False, loc='upper right', fontsize=9)
    sns.despine(ax=ax, left=True)
    if sozinho:
        titulo(titulo_grafico, subtitulo)
        salvar(nome_arquivo)
        plt.show()
    return tabela
