from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

labels_df = pd.read_csv("static/labels.csv")
labels = labels_df["breed"].unique().tolist()

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename != "":
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # ⚠️ 모델을 요청 시점에만 로드
            model = load_model("models/dog_breed_model.keras")

            img_array = preprocess_img(filepath)
            preds = model.predict(img_array)[0]
            pred_idx = np.argmax(preds)
            pred_label = labels[pred_idx]
            confidence = preds[pred_idx] * 100

            return render_template("result.html",
                                   user_image=filepath,
                                   dogcat_class=f"{pred_label} ({confidence:.1f}%)")
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
