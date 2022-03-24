import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Passo 1: Extração/Obtenção de Dados

df = pd.read_csv(r"\\endereço do arquivo no seu pc\\") #Lendo a base da dados
display(df) #exibindo a tabela

# Passo 2: Ajuste de Dados (Tratamento/Limpeza)

print(df.info()) #a base já está ajustada com os devidos tipos e valores

# Passo 3: Análise Exploratória

sns.pairplot(df) #gráfico de pontos 
plt.show() #exibindo gráfico
sns.heatmap(df.corr(), cmap="Wistia", annot=True) #gráfico de zona de calor
plt.show() #exibindo gráfico


# Passo 4: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)

# preparação dos dados para treinarmos o Modelo de Machine Learning (dados de teste e dados de treino)
x = df.drop('Vendas', axis=1)
y = df["Vendas"]
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.20)

#Escolha do modelo de regressão a ser utilizado ( Regressão Linear, RandomForest (Árvore de Decisão))
#Criando AI
treino_linear = LinearRegression()
treino_randon = RandomForestRegressor()

#treino AI
treino_linear.fit(x_treino, y_treino)
treino_randon.fit(x_treino, y_treino)

#Teste AI
teste_linear = treino_linear.predict(x_teste)
teste_randon = treino_randon.predict(x_teste)

r2_lin = metrics.r2_score(y_teste, teste_linear)
r2_rf= metrics.r2_score(y_teste, teste_randon)
print(f"R² da Regressão Linear: {r2_lin}")
print(f"R² da Random Forest: {r2_rf}")
mse_lin = metrics.mean_squared_error(y_teste, teste_linear)
mse_rf = metrics.mean_squared_error(y_teste, teste_randon)
print(f"MSE do Regressão Linear: {mse_lin}")
print(f"MSE do Random Forest: {mse_rf}")

#visualização gráfica dos resultados
df_resultado = pd.DataFrame()
df_resultado['Vendas Reais'] = y_teste
df_resultado['Previsão pelo RandomForest'] = teste_randon
df_resultado = df_resultado.reset_index(drop=True)
sns.lineplot(data=df_resultado)
plt.show()
display(df_resultado)

# Passo 5: Interpretação de Resultados

#Qual a importância de cada variável para as vendas
print(treino_randon.feature_importances_)
print("TV, Radio, Jornal\n")

#será que há investimento certo?
print(df[["Radio", "Jornal"]].sum())
