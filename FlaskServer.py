from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename

# Flask and CORS initialization
app = Flask(__name__)
CORS(app)

# Prepare image model
image_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Load models and tokenizers ahead of time
encoderCOCO = tf.keras.models.load_model('encoderCOCO')
decoderCOCO = tf.keras.models.load_model('decoderCOCO')
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizerCOCO = pickle.load(tokenizer_file)

encoderFlicker = tf.keras.models.load_model('encoder')
decoderFlicker = tf.keras.models.load_model('decoder')
with open('tokenizerfliker.pkl', 'rb') as tokenizer_file:
    tokenizerFlicker = pickle.load(tokenizer_file)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    return img, image_path

def evaluate(image, encoder, decoder, tokenizer):
    attention_plot = np.zeros((44, 49))
    hidden = tf.zeros((1, 512))
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(44):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)
    attention_plot = attention_plot[:len(result), :]

    return result, attention_plot

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify('No file part'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify('No selected file'), 400
    model_name = request.form.get('model')
    image_path = os.path.join('', secure_filename(file.filename))
    file.save(image_path)

    if model_name == 'COCO':
        encoder = encoderCOCO
        decoder = decoderCOCO
        tokenizer = tokenizerCOCO
    elif model_name == 'Flicker':
        encoder = encoderFlicker
        decoder = decoderFlicker
        tokenizer = tokenizerFlicker

    result, attention_plot = evaluate(image_path, encoder, decoder, tokenizer)
    result = [word for word in result if word != "<unk>"]
    result_final = ' '.join(result).rsplit(' ', 1)[0]
    return jsonify(result_final)

if __name__ == '__main__':
    app.run(port=7000)
