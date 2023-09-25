# API FLASK

Esse projeto visa a crição de uma API flask que permita fazer requisições à alguns modelos de inteligência artifical criados

## Endpoints
    /                    -> página inicial
    /sentimento/<frase>" -> GET Faz uma análise de sentimento do texto passado
    /cotacao/            -> POST faz uma predição de valor de uma casa dado um payload(estrutura : [tamanho, ano, garagem])       

    ** OBS: O cotacao e o sentimento necessitam de autentificação que se encontra no próprio código-fonte

## Bibliotecas utilizadas
- flask
- textblob
- pandas
- scikit-learn
- flask_basicauth

