from flask import Flask, request, render_template
import pickle


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    inputs = [
        float(request.args["glucose"]),
        float(request.args["blood"]),
        int(request.args["age"]),
        float(request.args["skin"]),
        int(request.args["pregnancies"]),
        float(request.args["insulin"]),
        float(request.args["bmi"]),
        float(request.args["dpf"])
    ]
    modello = request.args["model"]
    with app.open_resource(f"{modello}.bin", "rb") as f:
        model = pickle.load(f)
    output = model.predict([inputs])[0]
    response = "Modello: "+f"{modello}; Risultato: Diabetico" if output else f"{modello}; Risultato: Non diabetico"
    return render_template("predict.html", resp=response)
