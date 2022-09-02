from flask import Flask, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)


@app.route("/")
def hello_world():
    gen_model = tf.keras.models.load_model("./anime_generator_save")
    config = gen_model.get_config()
    noise_dim = config["layers"][0]["config"]["batch_input_shape"][-1]

    random_noise = [tf.random.normal(shape=(1, noise_dim), seed=seed) for seed in range(1)]
    random_noise = np.asarray(random_noise).reshape(1, noise_dim)

    imgs = gen_model(random_noise, training=False)
    pil_img = tf.keras.preprocessing.image.array_to_img(imgs[0])
    pil_img.save(f"./static/img.png")
    return render_template("index.html", img="./static/img.png")
    # return f"<p>Hello, World! {noise_dim}</p> <img src={pil_img}>"
