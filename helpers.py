import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# Load resources within a function to make it independent
def load_resources(model_file, labels_file):
    model = load_model(model_file, compile=False)
    with open(labels_file, "r") as file:
        labels = file.read().splitlines()
    return model, labels


def predict(image, model, labels):
    # Preprocess the image
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1

    # Prepare data for prediction
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    predictions = model.predict(data)

    # Extract prediction results
    index = np.argmax(predictions)
    class_name = labels[index]
    confidence_score = predictions[0][index]

    return class_name, confidence_score
