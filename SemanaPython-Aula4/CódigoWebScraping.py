# importar as bibliotecas
from selenium import webdriver
from selenium.webdriver.common.keys import Keys 
import pandas as pd

# 1º passo - abrir o navegador
chrome = webdriver.Chrome()

#se quiser utulizar com o chrome escondido
# from selenium.webdriver.chrome.options import Options -> importando biblioteca
# chrome_options = Options() 
# chrome_options.headless = True -> escondido = true
# nav = webdriver.Chrome(options=chrome_options)

# 2º passo - pesquisar as cotações no navegador

#dolar
chrome.get("https://www.google.com.br/")
chrome.find_element_by_xpath("/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input").send_keys("Cotação dólar")
chrome.find_element_by_xpath("/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input").send_keys(Keys.ENTER)
dolar = chrome.find_element_by_xpath('//*[@id="knowledge-currency__updatable-data-column"]/div[1]/div[2]/span[1]').get_attribute("data-value")

display("O valor do dólar é: "+dolar)

#euro
chrome.get("https://www.google.com.br/")
chrome.find_element_by_xpath("/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input").send_keys("Cotação Euro")
chrome.find_element_by_xpath("/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input").send_keys(Keys.ENTER)
euro = chrome.find_element_by_xpath('//*[@id="knowledge-currency__updatable-data-column"]/div[1]/div[2]/span[1]').get_attribute("data-value")

display("O valor do euro é: "+euro)

#ouro
chrome.get("https://www.melhorcambio.com/ouro-hoje")
ouro = chrome.find_element_by_xpath('//*[@id="comercial"]').get_attribute("value")
ouro = ouro.replace(",", ".")

display("O valor do ouro é: "+ouro)

#sair do navegador
chrome.quit()

# 3º passo - importar a base de dados

df = pd.read_excel(r"\\endereço do arquivo no seu pc\\")
display(df)

# 4º passo - atualizar os preços na base e reaizar o cálculo dos preços

#atualizando os preços
df.loc[df["Moeda"]=="Dólar", "Cotação"] = float(dolar)
df.loc[df["Moeda"]=="Euro", "Cotação"] = float(euro)
df.loc[df["Moeda"]=="Ouro", "Cotação"] = float(ouro)

#realizando os cáculos
df["Preço Base Reais"] = df["Cotação"] * df["Preço Base Original"] 
df["Preço Final"] = df["Ajuste"] * df["Preço Base Reais"] 

#formatando para 2 casa decimais
df["Preço Base Reais"] = df["Preço Base Reais"].map("{:.2f}".format)
df["Preço Final"] = df["Preço Final"].map("{:.2f}".format)

display(df)

# 5º passo - exporta a base de dados com os valores atualizados 

df.to_excel("Produtos_atualizados.xlsx", index=False)
