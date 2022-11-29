# Importing Libraries
import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_VMODULE"] = "gpu_process_state=10,gpu_cudamallocasync_allocator=10"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import jsonpickle
import cv2 as cv

app = Flask(__name__)
CORS(app)

tf.config.run_functions_eagerly(True)

# Declaring variables

STYLE_LAYERS = [
    ('block_1_project', 1.0),
    ('block_4_project', 0.8),
    ('block_8_project', 0.7),
    ('block_10_project', 0.2),
    ('block_12_project', 0.1)
]

CONTENT_LAYER = [
    ('block_14_project', 1),
]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)


def content_cost(content_image_output, generated_image_output):
    m, nh, nw, nc = content_image_output.get_shape().as_list()

    ac = tf.reshape(content_image_output, shape=[m, -1, nc])
    ag = tf.reshape(generated_image_output, shape=[m, -1, nc])

    j_content = tf.divide(tf.reduce_sum(tf.square(tf.subtract(ac, ag))), (4.0 * nh * nw * nc))

    return j_content


def gram_matrix(a):
    g = tf.matmul(tf.transpose(a), a)
    return g


def style_layer_cost(style_output_layer, generated_output_layer):
    m, nh, nw, nc = style_output_layer.get_shape().as_list()

    a_s = tf.reshape(style_output_layer, shape=[-1, nc])
    a_g = tf.reshape(generated_output_layer, shape=[-1, nc])

    gs = gram_matrix(a_s)
    gg = gram_matrix(a_g)

    j_style_layer = tf.divide(tf.reduce_sum(tf.square(tf.subtract(gs, gg))), (4.0 * (nh * nw * nc) ** 2))

    return j_style_layer


def style_cost(style_image_output, generated_image_output, layers=STYLE_LAYERS):
    a_s = style_image_output[:-1]
    a_g = generated_image_output[:-1]

    j_style = 0

    for i, weight in zip(range(len(a_s)), layers):
        j_style_layer = style_layer_cost(a_s[i], a_g[i])
        j_style += weight[1] * j_style_layer

    return j_style


@tf.function()
def total_cost(j_content, j_style, alpha=4.0, beta=10.0):
    return alpha * j_content + beta * j_style


def construct_model(layers, size):
    temp = tf.keras.applications.MobileNetV2(include_top=False,
                                             input_shape=size + (3,),
                                             weights='imagenet')
    temp.trainable = False
    output = [temp.get_layer(layer[0]).output for layer in layers]
    temp_model = tf.keras.Model([temp.input], output)
    return temp_model


def clip(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)


@tf.function()
def train(model, content_image, style_image, generated_image, alpha=1, beta=4):
    with tf.GradientTape() as tape:
        ag = model(generated_image)

        j_content = content_cost(content_image[-1], ag[-1])
        j_style = style_cost(style_image, ag)

        j = total_cost(j_content, j_style, alpha, beta)

    gradient = tape.gradient(j, generated_image)
    optimizer.apply_gradients([(gradient, generated_image)])
    generated_image.assign(clip(generated_image))
    return j


def generate(size, content_image_location, style_image_location, output_location, epochs, ith, alpha, beta):
    model = construct_model(STYLE_LAYERS + CONTENT_LAYER, size)

    c_image = np.array(Image.open(content_image_location).resize((size[1], size[0])))
    print(c_image.shape)
    c_image = tf.constant(np.reshape(c_image, ((1,) + c_image.shape)))

    s_image = np.array(Image.open(style_image_location).resize((size[1], size[0])))
    s_image = tf.constant(np.reshape(s_image, ((1,) + s_image.shape)))

    preprocessed_content = tf.Variable(tf.image.convert_image_dtype(c_image, tf.float32))
    a_c = model(preprocessed_content)

    preprocessed_style = tf.Variable(tf.image.convert_image_dtype(s_image, tf.float32))
    a_s = model(preprocessed_style)

    generated_image = tf.Variable(tf.image.convert_image_dtype(c_image, tf.float32))

    for i in range(epochs):
        train(model, a_c, a_s, generated_image, alpha, beta)
        if i % 10 == 0:
            print(f"{i} - epochs")

    image = tensor_to_image(generated_image)
    image.save(f"{output_location}/result.jpg")


def make_image():
    size = (750, 1000)

    content_image_location = "content/2.jpg"
    style_image_location = "style/3.jpg"
    output_location = "output/present"

    epochs = 1000
    ith = 10

    alpha = 4
    beta = 10

    generate(size, content_image_location, style_image_location, output_location, epochs, ith, alpha, beta)


def super_res():
    super_res = cv.dnn_superres.DnnSuperResImpl_create()
    super_res.readModel('FSRCNN_x2.pb')
    super_res.setModel('fsrcnn', 2)
    image = cv.imread("output/present/result.jpg")
    res = super_res.upsample(image)
    cv.imwrite("output/present/result.jpg", res)


@app.route("/api/compute/", methods=['POST'])
def compute():
    files = request.files.getlist('files')

    for file in files:
        temp = np.fromstring(file.read(), np.uint8)
        temp = cv.imdecode(temp, cv.IMREAD_UNCHANGED)
        cv.imwrite("content/2.jpg", temp)

    make_image()
    super_res()

    response = {"result": "Done"}

    res = jsonpickle.encode(response)

    return Response(response=res, status=200, mimetype="application/json")


@app.route("/home", methods=['GET'])
def get():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(host='192.168.121.22', port=5000)
