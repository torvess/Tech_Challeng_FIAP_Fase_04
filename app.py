import streamlit as st
import pandas as pd
import joblib
from joblib import load
from utils import ModelSeasonalNaive

st.set_page_config(page_title="Tech Challeng - Preços Petróleo", layout='wide')

st.title('Tech Challeng Fase 04 - Petróleo Brent')

tab1, tab2, tab3, tab4 = st.tabs(['Introdução', 'Análise Exploratória', 'Dashboard Analítico', 'Previsão de Valores por Data'])

with tab1:
    st.write('<b>Grupo:</b> Andre Antonio Campos, Clayton Gonçalves dos Santos, Debora Fabiana Pascoarelli, Igor Torves e Tamires Cristofani Suhadolnik', unsafe_allow_html=True)

    st.header('Sobre o Tech Challeng')

    texto = """
    ### Objetivos deste Challeng: 

    - **1**: Criação de WebScraping para extração dos dados no [site do IPEA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view).
    - **2**: Criação de um dashboard interativo utilizando power bi.
    - **3**: Análise exploratória dos dados utilizando o dashboard.
    - **4**: Desenvolvimento de modelo para previsão de preços diários utilizando a série temporal disponibilizada.
    - **5**: Construção de um site para exibir o dashboard e o deploy do modelo de Machine Learning.

    """

    st.markdown(texto)

with tab2:
    st.header('Análise Exploratória\n')

    st.subheader('Dados de Produção')

    st.markdown(
        """
        Analisando a serie no decorrer do tempo, tivemos tendências contrarias em períodos diferentes.

        * Entre 2005 e 2009 forte tendência de aumento de preços seguido por uma queda brusca
        * Entre 2010 e 2012 novo aumento seguido por 3 anos sem queda ou novos aumentos
        * 2015 novos aumentos que seguiram até 2020
        * Entre 2020 e 2023 nova tendência de aumento seguido por queda
        """)

    st.image('Midias\gaf1_serie_temporal.png')

    st.markdown(
        """
        Análise dos Preços do Barril de Petróleo Brent (1987-2023)

        * 1987: O preço médio ficou abaixo de 20 dólares, com a presença de alguns outliers.

        * 1988: O preço continuou abaixo de 20 dólares, levemente inferior ao ano anterior, sem outliers.

        * 1989: A média do preço permaneceu abaixo de 20 dólares, com um aumento observável e alguns outliers.

        * 1990: O preço médio atingiu 20 dólares, com um aumento significativo, sem outliers.

        * 1991: O preço médio permaneceu em 20 dólares, com um aumento significativo e a presença de alguns outliers.

        * 1992: A média manteve-se em 20 dólares, com aumento significativo, sem outliers.

        * 1993: O preço ficou abaixo de 20 dólares, sem outliers.

        * 1994: O preço manteve-se abaixo de 20 dólares, sem outliers.

        * 1995: Continuou abaixo de 20 dólares, sem outliers.

        * 1996: O preço médio subiu para 20 dólares, com um aumento significativo, sem outliers.

        * 1997: O preço caiu novamente para abaixo de 20 dólares, com a presença de alguns outliers.

        * 1998: Houve uma redução considerável, permanecendo abaixo de 20 dólares, sem outliers.

        * 1999: A média do preço ficou abaixo de 20 dólares, mas houve um aumento prolongado, sem outliers.

        * 2000: O preço mínimo foi ligeiramente acima de 20 dólares, com um aumento considerável nos preços, sem outliers.

        * 2001: O preço mínimo caiu para abaixo de 20 dólares, com valores inferiores ao ano anterior, sem outliers.

        * 2002: O preço médio aproximou-se de 30 dólares, semelhante ao ano anterior, mas apresentou outliers inferiores, indicando uma anomalia.

        * 2003: O preço subiu, com alguns outliers superiores ao valor máximo.

        * 2004: Aumento significativo, ultrapassando 40 dólares, sem outliers.

        * 2005: Continuação do aumento significativo, ultrapassando 60 dólares, sem outliers.

        * 2006: Aumento significativo, ultrapassando 60 dólares, sem outliers.

        * 2007: Aumento significativo, ultrapassando 80 dólares, sem outliers.

        * **2008: O preço teve um aumento significativo, mas houve uma grande discrepância entre a média e os valores mínimo e máximo, com outliers no valor mínimo. Sugere-se investigar as causas desse comportamento.**

        * 2009: A queda drástica levou o preço do barril abaixo de 80 dólares, sem outliers.

        * 2010: Pequeno aumento, ultrapassando 100 dólares, sem outliers.

        * 2011: Aumento significativo, ultrapassando 120 dólares, com outliers no valor mínimo.

        * 2012: Aumento significativo, ultrapassando 120 dólares, com outliers tanto no valor mínimo quanto no máximo.

        * 2013: Aumento significativo, ultrapassando 60 dólares, sem outliers.

        * **2014: Pequena queda no preço, com alguns outliers abaixo de 60 dólares. Recomenda-se investigar as causas.**

        * **2015: Queda drástica, com valores variando entre 35 e 65 dólares, sem outliers. Investigação das causas é sugerida.**

        * 2016: Pequena queda no preço, sem outliers.

        * 2017: Pequeno aumento, com alguns outliers acima do valor máximo.

        * 2018: Novo aumento, ultrapassando 120 dólares, com outliers no valor mínimo.

        * 2019: Preço abaixo de 80 dólares, sem outliers.

        * **2020: A variação foi extrema, com preços entre 10 e 70 dólares, apresentando outliers nos valores mínimo e máximo. Recomenda-se investigar as causas.**

        * 2021: Aumento no preço, indicando uma normalização, com outliers insignificativos.

        * 2022: Aumento considerável em relação ao ano anterior, variando entre 75 e 135 dólares, sem outliers.

        * 2023: Queda no preço, variando entre 75 e 95 dólares, sem outliers.
        """)
    
    st.image('Midias\gaf2_boxplot_preco_por_ano.png')

    st.markdown(
        """
        *Crescimento da Produção de Petróleo: A produção de petróleo aumentou de forma significativa desde o início dos anos 2000, estabilizando-se em torno de 2010. Esse crescimento reflete avanços na capacidade de produção, possivelmente devido a novas descobertas, avanços tecnológicos, ou investimentos na infraestrutura de extração de petróleo.*

        *Volatilidade dos Preços do Petróleo: A linha laranja que representa a média dos preços anuais do petróleo mostra uma grande volatilidade. Os preços subiram acentuadamente até cerca de 2008, atingindo um pico, seguido por uma queda acentuada. Depois de 2010, os preços continuaram a flutuar, com uma queda significativa por volta de 2015 e outra em torno de 2020, talvez relacionada a crises econômicas ou eventos globais (como a pandemia de COVID-19 em 2020).*
        """)
    
    st.image('Midias\gaf_3_producao_preco_por_ano.png')

    st.markdown(
        """
        *O gráfico da produção anual de barris de petróleo ao longo dos anos de 2000 a 2023, acompanhada da taxa de crescimento percentual anual. A análise dos dados oferece insights importantes sobre as flutuações no setor de petróleo, bem como sobre o impacto de eventos globais específicos na produção.*

        *Estabilidade Inicial (2000–2002): Entre 2000 e 2002, a produção de petróleo manteve-se estável, com oscilações muito pequenas, como o crescimento marginal de 0,04% em 2001 e uma leve queda de 0,26% em 2002.*

        *Período de Crescimento Moderado (2003–2008): De 2003 a 2008, a produção apresentou um crescimento constante, especialmente notável em 2004 (6,15%), refletindo um aumento na demanda global impulsionado pelo crescimento econômico em várias regiões, especialmente na Ásia.*

        *Crise Econômica Global (2009): Em 2009, observa-se uma queda de -1,05% na produção, refletindo o impacto da crise econômica global de 2008-2009, que reduziu a demanda e afetou a produção e os preços do petróleo.*

        *Recuperação e Crescimento (2010–2019): Após a crise, a produção volta a crescer, apresentando valores consistentes de aumento na maioria dos anos, com destaques como 2015 (3,28%) e 2018 (2,92%). Esse período reflete o fortalecimento da economia global e a recuperação da demanda por petróleo.*

        *Impacto da Pandemia (2020): Em 2020, há uma queda expressiva de -6,48% na produção, a maior de toda a série, devido à pandemia de COVID-19, que levou a uma redução drástica na demanda por petróleo e no consumo global de energia.*

        *Recuperação Pós-Pandemia (2021–2023): A partir de 2021, a produção retoma um crescimento moderado. Em 2022, destaca-se uma recuperação significativa de 4,69%, com a demanda global voltando aos níveis pré-pandêmicos. Em 2023, o crescimento continua, embora em um ritmo mais moderado (1,9%).*
        """)
    
    st.image('Midias\gaf4_crescimento_barris_por_ano.png')

    st.markdown(
        """
        *Os dados apresentados mostram a produção de petróleo em barris por ano para os principais países produtores, destacando o domínio de algumas nações no setor.*

        *Estados Unidos lidera a produção global com 307.914 milhões de barris anuais, refletindo sua posição consolidada como principal produtor mundial. A capacidade de extração dos EUA se apoia em avanços tecnológicos, como o fraturamento hidráulico (fracking), que aumentou significativamente a produção nas últimas décadas.*

        *Arábia Saudita ocupa o segundo lugar, com 259.531 milhões de barris, impulsionada por suas vastas reservas no Oriente Médio e pelo papel central que desempenha na Organização dos Países Exportadores de Petróleo (OPEP).*

        *Rússia, em terceiro, com 241.077 milhões de barris, também desempenha um papel fundamental no mercado global de petróleo, fornecendo uma quantidade significativa para a Europa e a Ásia.*

        *China e Canadá produzem 105.279 milhões e 97.562 milhões de barris, respectivamente. A China, embora seja um grande consumidor de petróleo, possui também uma produção significativa. O Canadá, por sua vez, conta com reservas consideráveis de petróleo, especialmente nas areias betuminosas de Alberta.*

        *Outros países notáveis incluem o Irã (94.040 milhões), que enfrenta desafios devido a sanções econômicas, e os Emirados Árabes Unidos (79.722 milhões), que são conhecidos por suas reservas abundantes e política energética de expansão.*

        *Iraque (75.532 milhões), Brasil (63.811 milhões), e Kuwait (63.108 milhões) completam a lista, cada um com um papel relevante na produção global. O Brasil, o único país da *América Latina na lista, destaca-se pela exploração de petróleo em águas profundas, como na região do pré-sal.*

        *Esses dados sublinham a importância estratégica do petróleo para cada uma dessas economias e evidenciam as diferenças na capacidade de produção entre os países, influenciando o mercado energético global.*
        """)
    
    st.image('Midias\gaf5_top_paises_producao.png')

    st.subheader('Dados Consumo de Energia')

    st.markdown(
        """
        * *O consumo médio de petróleo, representado pelas barras, parece relativamente estável ao longo dos anos, com algumas flutuações, mas sem uma tendência de crescimento ou declínio acentuado. Isso indica que a demanda por petróleo estável ao longo do tempo, apesar de eventuais variações.*

        * *A linha laranja mostra que a produção de petróleo passou por várias oscilações, com uma queda acentuada em meados dos anos 2000 e um aumento significativo a partir de 2010.
        Esse aumento recente na produção pode estar associado a novas tecnologias de extração, como o fracking, ou à exploração de novos campos petrolíferos.*

        * *O gráfico destaca a relação complexa entre produção e consumo de petróleo, onde fatores externos e avanços tecnológicos parecem ter papel significativo. A análise dessas tendências ajuda a entender a segurança energética e a dependência de importações, além de fornecer insights sobre a resiliência da produção frente a demandas flutuantes.*
        """)
    
    st.image('Midias\gaf6_producao_consumo_anual.png')

    st.markdown('Top 10 países com maior consumo de energia')

    st.image('Midias\gaf7_top_paises_maior_consumo_energia.png')

    st.subheader('PIB Mundial')

    st.image('Midias\gaf8_crescimento_pib.png')

    st.subheader('Mercado de Ações')

    st.markdown(
        """
        Gráfico de evolução mensal (1999-2023):

        * Este gráfico apresenta os valores mensais do índice S&P 500 em USD.
        * Identifica-se uma queda acentuada por volta de 2008, marcada pela linha vermelha, que coincide com a crise financeira global. Após esse período, nota-se uma recuperação consistente e uma tendência de alta ao longo da última década.
        * Picos e quedas menores refletem eventos econômicos ou geopolíticos que podem ter impactado o mercado.
        """)
    
    st.image('Midias\gaf9_desempenho_acoes.png')

    st.markdown(
        """
        Gráfico de evolução anual (2000-2023):

        * Este gráfico apresenta uma visão mais simplificada dos valores médios anuais, mostrando tendências claras ao longo do tempo.
        * Observa-se que a crise de 2008 provocou uma queda abrupta, seguida por uma recuperação mais estável após 2010.
        * Outra linha vermelha em 2020 destaca uma queda relacionada à pandemia de COVID-19, seguida por um crescimento acelerado nos anos seguintes.
        """)
    
    st.image('Midias\gaf10_desempenho_acoes2.png')

    st.subheader('Taxa de Juros')

    st.image('Midias\gaf11_taxa_juros.png')

    st.markdown(
        """
        * ### Bank Capital to Assets Ratio (Razão de Capital Bancário para Ativos):
        Reflete a saúde financeira dos bancos, mostrando a quantidade de capital que eles possuem em relação aos seus ativos. Serve para avaliar a solidez financeira do banco e sua capacidade de absorver perdas.
        """)
    
    st.image('Midias\gaf12_juros_emprestimo.png')

    st.markdown(
        """
        * ### Real Interest Rate (Taxa de Juros Real):
        É a taxa de juros ajustada pela inflação, refletindo o verdadeiro custo do dinheiro e o retorno real de um investimento ou empréstimo.
        """
    )

    st.image('Midias\gaf13_taxa_juros_razao_capital_bancario_pais.png')

    st.subheader('Análise Crise Financeira 2008')

    st.markdown(
        """
        A crise de 2008, também conhecida como a Grande Recessão, foi uma grave crise financeira global que começou nos Estados Unidos e rapidamente se espalhou pelo mundo. Ela foi desencadeada por uma bolha imobiliária nos EUA, alimentada por empréstimos hipotecários de alto risco (subprime) e práticas financeiras pouco reguladas.

        *Consequências:*
        * Recessão Global: A economia mundial encolheu, com empresas falindo e milhões de pessoas perdendo empregos e residências.
        * Crise Bancária: Bancos ao redor do mundo enfrentaram colapsos, forçando governos a intervir com pacotes de resgate.
        * Queda nos Preços do Petróleo: A desaceleração econômica reduziu a demanda por petróleo, provocando uma queda significativa nos preços.
        """)
    
    st.image('Midias\gaf14_crise_2008.png')

    st.image('Midias\gaf15_desempenho_acoes_2007_a_2009.png')

    st.markdown(
        """
        A Crise Financeira de 2008, também conhecida como a Grande Recessão, gerou mudanças econômicas globais drásticas. Durante esse período, vários indicadores-chave, como o preço do petróleo, o PIB global e as taxas de juros, passaram por variações significativas.

        1. Preço do Petróleo
        O preço do petróleo teve uma alta abrupta no primeiro semestre de 2008, chegando a mais de 140 USD por barril. No entanto, devido à desaceleração econômica global e a queda na demanda, o preço despencou mais de 70% até o final do ano, resultando em uma variação de 168.26% quando comparado ao valor antes da crise. A volatilidade reflete tanto a especulação sobre a escassez de petróleo quanto a reação à crise financeira.

        2. Produto Interno Bruto (PIB)
        O PIB global sofreu uma retração acentuada, com uma leve variação de 0.16% em termos reais. Embora a crise tenha sido particularmente severa para economias desenvolvidas, os países emergentes conseguiram mitigar, em parte, os impactos negativos. Esse leve aumento no PIB real reflete uma desaceleração generalizada, mas não uma recessão extrema, especialmente nas economias em crescimento.

        3. Taxa de Juros
        Para combater a desaceleração econômica, os bancos centrais, como o Federal Reserve dos EUA, reduziram drasticamente as taxas de juros. Essa medida reflete uma política monetária expansionista, mas a variação de -0.31% nos juros reais indica uma diminuição gradual em relação aos níveis mais altos anteriores à crise.

        4. S&P 500
        O índice S&P 500, refletindo o desempenho do mercado de ações dos EUA, sofreu uma queda de -26.81% no preço real do petróleo durante a crise de 2008. Esse declínio foi um reflexo do pânico e da instabilidade nos mercados financeiros, que afetaram negativamente a confiança dos investidores e o valor das ações.

        Insights Principais:
        * Preço do Petróleo: A crise causou um pico e subsequente queda acentuada nos preços, demonstrando a vulnerabilidade do mercado de petróleo a choques econômicos.
        * PIB Global: O crescimento do PIB global foi moderadamente afetado, com economias desenvolvidas sendo as mais impactadas, enquanto as emergentes conseguiram resistir.
        * Taxas de Juros: As políticas de redução das taxas de juros foram agressivas, refletindo os esforços dos bancos centrais para estimular a economia e evitar uma recessão mais profunda.
        * Mercados Financeiros (S&P 500): O mercado de ações experimentou uma das maiores quedas da história, refletindo a crise de confiança nos mercados financeiros.
        """)
    
    st.subheader('Impacto da COVID-19 (2019, 2020 e 2021) no Preço do Petróleo')

    st.image('Midias\gaf16_impacto_covid.png')

    st.image('Midias\gaf17_desempenho_acoes_tempos_covid.png')

    st.markdown(
        """
        *O preço do petróleo sofreu uma grande queda no início de 2020, caindo de cerca de $60 para menos de $20 por barril até abril, devido aos lockdowns globais impostos pela pandemia de COVID-19, que reduziram drasticamente a demanda por combustíveis (UNC Global Affairs). Em abril, a crise se agravou, e o excesso de oferta de petróleo, aliado à escassez de armazenamento, levou a preços negativos em alguns mercados dos EUA.*

        *A partir do segundo semestre de 2020, o mercado de petróleo iniciou uma recuperação lenta. No final do ano, o preço se estabilizou em torno de $50 por barril, refletindo a expectativa de recuperação econômica com o início das campanhas de vacinação contra a COVID-19 e a adaptação global às novas condições de mercado.*
        """)
    
    st.image('Midias\gaf18_pib_tempos_covid.png')

    st.markdown(
        """
        *2019-2020: A alta na taxa de juros real pode ser atribuída a expectativas econômicas pré-pandemia, com os países já enfrentando pressões de inflação e crescimento.*

        *2020-2022: A queda drástica nas taxas de juros reais durante a pandemia pode ser explicada pelas políticas monetárias expansionistas adotadas para combater os efeitos econômicos da COVID-19. O banco central reduziu as taxas de juros para estimular a economia, facilitar o acesso ao crédito e mitigar a recessão econômica causada pela pandemia. A política de juros baixos também pode ter sido usada para combater a deflação, dada a queda na demanda agregada durante o período de confinamento e crise econômica.*
        """)
    
    st.image('Midias\gaf19_taxa_juros_tempos_covid.png')

    st.markdown(
        """
        *Em 2019, os preços do petróleo se mantiveram relativamente estáveis ao longo do ano, sem grandes variações. No entanto, em 2020, houve uma queda significativa nos preços devido aos impactos da pandemia de COVID-19, que causaram uma redução drástica na demanda global. Em 2021, com a recuperação econômica e a reabertura de mercados, o mercado de petróleo começou a se recuperar, e os preços começaram a subir novamente.*
        """
    )

    st.image('Midias\gaf20_preco_mensal_petroleo_tempos_covid.png')

    st.markdown(
        """
        #### Conclusão:

        * *Impacto da COVID-19: Em todos os gráficos, especialmente no primeiro e segundo, a queda drástica em 2020 confirma o impacto da pandemia no mercado de petróleo. Com a desaceleração global, a demanda caiu abruptamente, causando uma queda histórica nos preços.*

        * *Recuperação Gradual: A recuperação dos preços é visível após o pior momento da pandemia, com o aumento gradual da demanda conforme as economias reabriram e as viagens voltaram a ocorrer. No entanto, essa recuperação não foi rápida, pois o mercado de petróleo enfrentou desafios de oferta e demanda.*

        * *Alta Volatilidade: O terceiro gráfico destaca uma alta volatilidade durante o período mais crítico da pandemia. As grandes flutuações de preço foram provocadas pela incerteza no mercado e pela incapacidade de prever o retorno da demanda, além de choques de oferta devido a decisões da OPEP e disputas entre grandes produtores.*

        * *Tendências de Longo Prazo e Sazonalidade: Os gráficos de tendências mais amplas (segundo e quarto) mostram uma queda prolongada seguida de uma recuperação lenta. Isso aponta para um mercado que, embora fortemente impactado no curto prazo, tem uma capacidade de recuperação cíclica e sazonal conforme as condições macroeconômicas se estabilizam.*

        Fontes: 
        * Bureau of Labor Statistics — From the barrel to the pump: the impact of the COVID-19 pandemic on prices for petroleum products : Monthly Labor Review: U.S. Bureau of Labor Statistics - bls.gov
        * EIA — OPEC+ agreement to reduce production contributes to global oil market rebalancing - U.S. Energy Information Administration (EIA) - eia.gov
        * UNC Global Affairs — How the COVID-19 Pandemic Plunged Global Oil Prices - UNC Global Affairs - global.unc.edu
        """)


with tab3:
    with st.container():
        st.header("Preços do Petróleo no Mundo")
        st.write("Informações sobre preço de petróleo dos principais países do mundo (em Dólar).")
        st.write("Período: 1987 a 2023")

    # URL do relatório Power BI para incorporar
    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiZTZiNzFhY2YtMjI2YS00MGUwLWJhY2ItYjhlMGJmYTc5YmE5IiwidCI6ImRlOTgwY2Y4LWYwYzctNGFlZC1iNjc2LTJlOTlkNjg2YzAzMyJ9" 
                    
    # Tamanho do iframe
    iframe_width = None
    iframe_height = 700

    # Exibir o relatório Power BI
    st.components.v1.iframe(power_bi_url, width=iframe_width, height=iframe_height)

    #############################################################################################################################

    st.subheader("Análise descritiva referente aos indicadores do Dashboard acima:")

    ## 1
    st.subheader("1. Tendência Anual e Longo Prazo")
    st.write("Observa-se uma média anual crescente até 2022, com picos significativos em anos de alta demanda energética e eventos críticos, como 2021 e 2022, onde o preço médio anual foi de aproximadamente 70 e 100 (dólar), respectivamente. Esses aumentos estão correlacionados com a recuperação pós-pandemia e tensões geopolíticas.")

    ## 2
    st.subheader("2. Maiores Aumentos Anuais no Preço do Brent")
    st.write("* **2021**: aumento de 69,87%, refletindo o retorno da demanda pós-COVID-19.")
    st.write("* **2000**: aumento de 60,10%, relacionado a tensões no Oriente Médio e aumento da demanda.")
    st.write("* Outros picos em 2011 e 2008 indicam períodos de alta demanda e instabilidade política, como a crise financeira global.")

    ## 3
    st.subheader("3. Maiores Quedas Anuais no Preço do Brent")
    st.write("* **2015**: queda de 47,15%, refletindo excesso de oferta e desaceleração econômica.")
    st.write("* **2009**: queda de 37,13%, diretamente relacionada à crise financeira global de 2008.")
    st.write("* Quedas em 2020 e 1998 mostram o impacto de crises econômicas e de demanda global.")

    ## 4
    st.subheader("4. Sazonalidade Mensal")
    st.write("* Os preços tendem a atingir o pico entre os meses de março a setembro, com preços médios mais altos em maio, julho e agosto, refletindo a alta demanda de verão no hemisfério norte")
    st.write("* Os menores preços médios são observados nos meses de dezembro e janeiro, alinhando-se ao fim de ano e menor demanda de energia.")

    ## Volatilidade

    st.subheader("Análise da Volatilidade por Década e Impactos Globais")

    st.subheader("Década de 1990")
    st.write("* O início dos anos 1990 foi marcado pela Guerra do Golfo (1990-1991), que gerou um aumento na volatilidade, com os preços reagindo rapidamente à incerteza sobre a produção no Oriente Médio.")
    st.write("* Em 1998, uma combinação de excesso de oferta e a crise financeira asiática resultou em quedas acentuadas no preço do Brent, levando a uma volatilidade alta, que só foi estabilizada pelo esforço de controle da oferta pela OPEP.")

    st.subheader("Década de 2000")
    st.write("* Nos primeiros anos, os preços começaram a se recuperar em resposta ao aumento da demanda global, particularmente da China, e aos receios de restrições de oferta.")
    st.write("* Eventos como a invasão do Iraque em 2003 e as preocupações com a segurança das rotas de fornecimento impulsionaram a volatilidade.")
    st.write("* Em 2008, a crise financeira global trouxe uma volatilidade sem precedentes, com o preço do Brent caindo rapidamente após um pico histórico no início do ano, devido à queda brusca da demanda global e às incertezas econômicas.")

    st.subheader("Década de 2010")
    st.write("* A década começou com o Brent em alta volatilidade devido à Primavera Árabe e outras instabilidades no Oriente Médio e no Norte da África, que afetaram a produção de países como a Líbia.")
    st.write("* Em 2014-2015, o Brent passou por uma queda abrupta de preços, com o aumento da produção de petróleo de xisto nos Estados Unidos criando uma sobreoferta no mercado global, aumentando a volatilidade em resposta à desaceleração econômica na China e à redução da demanda global.")
    st.write("* A crise de preços em 2015 e 2016 fez com que a volatilidade permanecesse alta até que a OPEP e outros países produtores implementassem cortes de produção para estabilizar o mercado.")

    st.subheader("Década de 2020")
    st.write("* A pandemia de COVID-19 em 2020 causou uma queda histórica na demanda por petróleo, especialmente nos meses iniciais, resultando em um aumento dramático na volatilidade. O Brent chegou a preços mínimos históricos em função do excesso de estoque, dificuldades de armazenamento e incertezas sobre a recuperação econômica global.")
    st.write("* Em 2021, a demanda voltou a subir com a reabertura das economias, o que levou a uma rápida recuperação de preços e alta volatilidade devido à instabilidade na oferta, problemas logísticos e aumento dos custos de energia.")
    st.write("* Em 2022, a invasão da Ucrânia pela Rússia trouxe uma nova onda de incertezas no mercado de energia global, já que a Rússia é um dos maiores exportadores de petróleo. A volatilidade disparou com sanções, mudanças de rotas de suprimento e aumento da incerteza sobre a capacidade de fornecimento da Europa e da Ásia.")

    st.subheader("Considerações PIB x Preço do Petróleo Brent")
    st.write("Em períodos de crescimento econômico global sustentável e estável, como no início dos anos 2000, observa-se que o preço do petróleo aumenta gradativamente, acompanhando a demanda crescente. Esse comportamento pode indicar que o PIB global e o preço do petróleo compartilham uma relação positiva em condições econômicas saudáveis.")
    st.write("A volatilidade do preço do petróleo, devido a eventos geopolíticos, como conflitos no Oriente Médio ou sanções, pode afetar a estabilidade econômica, especialmente em países fortemente dependentes da exportação ou importação de petróleo. Esses fatores muitas vezes se refletem em taxas de crescimento do PIB mais instáveis.")

with tab4:
    st.header('Previsão de Preços por Data ⛽')

    date = st.date_input("Selecione uma data", pd.to_datetime('2024-01-01'), min_value=(pd.to_datetime('2024-01-01')))

    model_sn = joblib.load(r'Modelos/SeasonalNaive.joblib')

    if st.button('Executar Previsão'):
        resultado = model_sn.predic_SeasonalNaive(pd.to_datetime(date))

        st.write(f'O valor para <b>{date.strftime("%d/%m/%Y")}</b> é R$ <b>{resultado:.2f}</b>', unsafe_allow_html=True)
    