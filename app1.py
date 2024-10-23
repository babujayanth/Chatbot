import os
import json

from flask import Flask, render_template, request,jsonify

# from chat import get_response

from main import get_response
from main import predict_class


app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
data_file = os.path.join(basedir, 'data.json')



@app.get("/")

def index_get():
    return render_template("base.html")

# @app.post("/read")
# def readfile():
#     WORDS = []

#     with open(data_file, "r") as file:
#         for line in file.readlines():
#          WORDS.append(line.rstrip())
#     words = [word for word in WORDS ]
#     return jsonify(words)

# @app.post("/write")
# def writefile():
#     text=request.get_json().get("message")
#     with open(data_file, "a") as fo:
#         fo.write(text+ "\n")
#     return jsonify({"msg":"sces"})


@app.post("/predict")
def predict():
    text=request.get_json().get("message")
    print('a:',text)
    intents = json.loads(open('intents.json').read())
    ints = predict_class(text)
    print(ints)
    response = get_response(ints,intents)
    print('b:',response)
    message = {"answer":response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)