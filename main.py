from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

colunas = ["tamanho", "ano", "garagem"]
# Modelo
model = pickle.load(open("model.sav", "rb"))

#Facilita a busca do flask
app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = "Nat"
app.config["BASIC_AUTH_PASSWORD"] = "1234"

basic_auth = BasicAuth(app)

@app.route("/")
def home():
    return 'Minha primeira API.'

#a variável é cirada entre <>
@app.route("/sentimento/<frase>")
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang="pt_br", to="en")

    polaridade = tb_en.sentiment.polarity

    return f'Polaridade: {polaridade}'

@app.route("/cotacao/", methods=["POST"])
@basic_auth.required
def cotacao():

    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]

    preco = model.predict([dados_input])


    #Retorno como json
    return jsonify(preco=preco[0])

#executa novamente a partir de qualquer nova alteração
app.run(debug=True) 