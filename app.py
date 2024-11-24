import streamlit as st
import pandas as pd
import joblib
from joblib import load
from utils import ModelSeasonalNaive

st.set_page_config(page_title="Tech Challeng - Preços Petróleo", layout='wide')

st.title('Tech Challeng Fase 04 - Petróleo Brent')

tab1, tab2, tab3 = st.tabs(['Introdução', 'Dashboard Analítico', 'Previsão de Valores por Data'])

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

with tab3:
    st.header('Previsão de Preços por Data ⛽')

    date = st.date_input("Selecione uma data", min_value=(pd.to_datetime('2024-01-01')))

    model_sn = joblib.load(r'Modelos/SeasonalNaive.joblib')

    if st.button('Executar Previsão'):
        resultado = model_sn.predic_SeasonalNaive(pd.to_datetime(date))

        st.write(f'O valor para <b>{date.strftime("%d/%m/%Y")}</b> é R$ <b>{resultado:.2f}</b>', unsafe_allow_html=True)
    