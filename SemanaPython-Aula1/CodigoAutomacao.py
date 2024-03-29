import pyautogui as pa
import time as tm 
import pyperclip as py
import pandas as pd
import glob as gl
import os

#1 segundo de pausa entre os comandos e um alerta de inicio de código
pa.PAUSE = 1.5
pa.alert("A automação irá começar a rodar, por favor não mexa em nada até ela finalizar")

#iniciando da área de trabalho
# pa.press("winleft")
# pa.write("chrome")
# pa.press("enter")

#iniciando pelo chrome normal
pa.hotkey("ctrl", "t")

#se quiser na aba anônima
# pa.hotkey("ctrl", "shift", "n")
# pa.hotkey("alt", "tab")
# pa.hotkey("ctrl", "w")

#abrir o google drive com a planilha
link = "https://drive.google.com/drive/folders/1mhXZ3JPAnekXP_4vX7Z_sJj35VWqayaR?usp=sharing"
py.copy(link)
pa.hotkey("ctrl", "v")
pa.press("enter")
tm.sleep(4)

#Caso o email já esteja cadstado, abrir a base de dados e baixá-la
pa.click(338, 279, clicks=2)
pa.click(1088, 343)
pa.click(1154, 168)
tm.sleep(1)
pa.click(1016, 528)
tm.sleep(10)

#abrir a base de dados e baixá-la sem email cadastrado
# pa.click(198, 279)
# pa.click(927, 332)
# pa.click(1286, 95)
# tm.sleep(8)

#realizar o calculo dos indicadores na base da dados
tabela = pd.read_excel(r"\\endereço do arquivo no seu pc\\")
display(tabela)
faturamento = tabela["Valor Final"].sum()
qntd_produtos = tabela["Quantidade"].sum()

#fechar o dowload do navgador
pa.click(1347, 702)

#abrir o email 
pa.hotkey('ctrl', 't')
pa.write("mail.google.com")
pa.press('enter')
tm.sleep(3)

#Caso não tenha o email logado ainda, cadastrar na conta
# email = ("igormozetic04")
# py.copy(email)
# pa.hotkey("ctrl", "v")
# pa.press('enter')
# tm.sleep(10)

#Caso email já esteja logado, montar o email e enivar para a diretoria
pa.click(97, 169)
tm.sleep(2)
pa.write("igormozetic04+diretoria@gmail.com")
pa.press("tab")
pa.press("tab")
assunto = "Relatório de Vendas"
py.copy(assunto)
pa.hotkey("ctrl", "v")
pa.press("tab")
texto = f"""
Bom dia excelentíssimo chefe, 

Segue abaixo o relatório de Vendas dos indicadores faturamento e quantidade de produtos que foi solicitado:

O faturamento foi de: R${faturamento:,.2f}
A Quantidade de Produtos foi de: {qntd_produtos}
"""
py.copy(texto)
pa.hotkey("ctrl", "v")
pa.click(1141, 700)
pa.click(1172, 629)

#enivar o email
pa.hotkey('ctrl', 'enter')

#alerta que a automação acabou
pa.alert("Fim da Automação.")
