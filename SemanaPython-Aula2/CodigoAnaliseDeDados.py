# Importar as bibliotecas que serão utilizadas
import pandas as pd
import plotly.express as px

# Passo 1: Importar base de dados
df = pd.read_csv(r"\\endereço do arquivo no seu pc\\")

# Passo 2: Visualizar a base de dados
df = df.drop(columns=['Unnamed: 0', 'IDCliente', 'Codigo']) #removendo duas colunas inúteis para a análise
display(df)

# Passo 3: Realizar o tratamento dos dados
df["TotalGasto"] = pd.to_numeric(df["TotalGasto"], errors="coerce") #colocando a coluna TotalGasto para numérico
df["Aposentado"] = df["Aposentado"].astype(str)  #colocando a coluna Aposentado para objeto
df = df.dropna(how='all', axis=1) #remover coluna que está completamente vazia
df = df.dropna(how='any', axis=0) #remover linhas que não estão completas - poderai usar também "df = df.dropna"
print(df.info()) #exibindo as infos da df

# Passo 4: Visão geral da distribuição do churn/cancelamento
display(df["Churn"].value_counts()) #quantidade de cancelamentos e permanencias em números
display(df["Churn"].value_counts(normalize=True).map('{:,.1%}'.format)) #quantidade de cancelamentos e permanencias em % e formatado

# Passo 5: Analisar como cada característica do cliente impacta no indicador de churn

for coluna in df:
    tabela = px.histogram(df, x=coluna, color="Churn")
    tabela.show()


#Conclusão para o chefe
# - A diferença de cancelamento nos gêneros não tem muita diferença
# - Pouco menos que a metade dos aposentados canelaram nosso plano, enquanto os não aposentados, menos de 1/3 cancelaram
# - 1/3 dos solteiros cancelaram nosso plano, enquanto os casasdos pouco mias de 1/5
# - Quem não tem dependentes está cancelando muito mais dos que tem dependentes
# - A taxa de cancelamento nos 10 primeiros meses é superior aqueles que estão a mais tempo com o plano
# - Há muito mais cancelamentos naqueles que não contém os serviços que nós oferecemos
# - Contratos mensais tem alta taxa de cancelamento
# - fatura digital está cancelando mais
# - boleto eletrônico pior forma de pagamento
# - valores mensais variam muito, porém quanto mais caro, há maior cancelamento
# - Antes de gastarem R$1000,00 dentro da empresa a taxa de cancelamento é alta.
