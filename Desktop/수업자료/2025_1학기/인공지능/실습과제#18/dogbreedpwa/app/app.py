from flask import Flask, render_template, request
import os, numpy as np, pandas as pd, time, gc
import tflite_runtime.interpreter as tflite
from PIL import Image

app = Flask(__name__)
UPLOAD = "static/uploads"
os.makedirs(UPLOAD, exist_ok=True)

# ── 라벨
labels = pd.read_csv("static/labels.csv")["breed"].unique().tolist()

# ── TFLite 모델 로드 (메모리 30~70 MB)
interpreter = tflite.Interpreter(model_path="models/dog_breed_model.tflite")
interpreter.allocate_tensors()
input_index  = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224,224))
    arr = np.expand_dims(np.asarray(img, dtype=np.float32)/255.0, 0)
    return arr

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename:
            save = os.path.join(UPLOAD, f.filename)
            f.save(save)

            t0 = time.time()
            interpreter.set_tensor(input_index, preprocess(save))
            interpreter.invoke()
            preds = interpreter.get_tensor(output_index)[0]
            elapsed = round(time.time()-t0, 2)

            idx = int(np.argmax(preds))
            return render_template("result.html",
                                   user_image=save,
                                   dogcat_class=f"{labels[idx]} ({preds[idx]*100:.1f} %)",
                                   elapsed=elapsed)
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
