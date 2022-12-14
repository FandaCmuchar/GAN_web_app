from flask import Flask, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
gen_model = tf.keras.models.load_model("./static/swords_generator_save")
config = gen_model.get_config()
noise_dim = config["layers"][0]["config"]["batch_input_shape"][-1]


@app.route("/")
def hello_world():
    random_noise = [tf.random.normal(shape=(1, noise_dim), seed=seed) for seed in range(16)]
    random_noise = np.asarray(random_noise).reshape(16, noise_dim)

    imgs = gen_model(random_noise, training=False)
    img_names = []
    for i in range(len(imgs)):
        pil_img = tf.keras.preprocessing.image.array_to_img(imgs[i])
        pil_img = pil_img.resize((64, 64), Image.ANTIALIAS)
        pil_img.save(f"./static/img{i}.png")
        img_names.append(f"./static/img{i}.png")
    return render_template("index.html", img=img_names)
    # return f"<p>Hello, World! {noise_dim}</p> <img src={pil_img}>"
