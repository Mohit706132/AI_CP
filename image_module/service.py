"""Keras image inference service for the integrated Streamlit app."""

import io
from pathlib import Path

import numpy as np
from PIL import Image


IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 40.0

HIRDA_CLASSES = [str(index) for index in range(1, 9)]
DETECTOR_CLASSES = ["haritaki", "vidanga"]
VIDANGA_CLASSES = [
    "Vidanga_Species 1_Embelia_ribes",
    "Vidanga_Species 1_Embelia_tsjerium_cottam",
]

HIRDA_INFO = {
    "1": {"name": "Vijaya", "description": "Found in Vindhya region. Round-shaped fruit.", "uses": "Digestive disorders, fever, skin diseases, general tonic"},
    "2": {"name": "Rohini", "description": "Round-shaped fruit. Used for wound healing.", "uses": "Wound healing, skin infections, external applications"},
    "3": {"name": "Putana", "description": "Small fruit with large seed.", "uses": "External applications, skin care"},
    "4": {"name": "Amrita", "description": "Fleshy fruit. Best for Panchakarma.", "uses": "Panchakarma, detoxification, rejuvenation"},
    "5": {"name": "Abhaya", "description": "5-lined fruit. Effective for eye diseases.", "uses": "Eye diseases, digestive issues, constipation"},
    "6": {"name": "Jivanti", "description": "Yellow-colored. Considered best quality.", "uses": "Longevity, vitality, overall health"},
    "7": {"name": "Chetaki", "description": "3-lined fruit. Strong laxative properties.", "uses": "Constipation, laxative, bowel cleansing"},
    "8": {"name": "Kayastha", "description": "Ink-black colored. Used for hair and skin.", "uses": "Hair care, skin treatments, anti-aging"},
}

VIDANGA_INFO = {
    VIDANGA_CLASSES[0]: {"name": "Embelia ribes", "description": "True Vidanga. Small round berries, reddish-brown when ripe.", "uses": "Anthelmintic, digestive stimulant, anti-obesity, skin diseases"},
    VIDANGA_CLASSES[1]: {"name": "Embelia tsjerium-cottam", "description": "Substitute Vidanga species with distinct morphology.", "uses": "Substitute for Embelia ribes, anthelmintic properties, digestive aid"},
}


def _load_keras_models(models_dir: Path):
    import tensorflow as tf

    models_dir = Path(models_dir)

    def load(preferred, fallback):
        preferred_path = models_dir / preferred
        path = preferred_path if preferred_path.exists() else models_dir / fallback
        return tf.keras.models.load_model(path)

    return (
        load("plant_detector.keras", "plant_detector.keras"),
        load("hirda_best_model_finetune.keras", "hirda_best_model.keras"),
        load("vidanga_best_model_finetune.keras", "vidanga_best_model.keras"),
    )


def load_models(models_dir: Path):
    """Load detector and specialist models; intended for Streamlit caching."""
    return _load_keras_models(models_dir)


def preprocess(image_bytes: bytes):
    import tensorflow as tf

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    display_image = image.copy()
    image = image.resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return display_image, array


def _info(mapping, key, fallback):
    return mapping.get(key, {"name": fallback, "description": "", "uses": ""})


def predict(image_bytes: bytes, models):
    display_image, image_array = preprocess(image_bytes)
    detector_model, hirda_model, vidanga_model = models

    detector_probs = detector_model.predict(image_array, verbose=0)[0]
    detector_index = int(np.argmax(detector_probs))
    plant_type = DETECTOR_CLASSES[detector_index]
    plant_confidence = float(detector_probs[detector_index]) * 100

    if plant_type == "haritaki":
        probabilities = hirda_model.predict(image_array, verbose=0)[0]
        classes = HIRDA_CLASSES
        info_map = HIRDA_INFO
        plant_label = "Haritaki (Terminalia chebula)"
    else:
        probabilities = vidanga_model.predict(image_array, verbose=0)[0]
        classes = VIDANGA_CLASSES
        info_map = VIDANGA_INFO
        plant_label = "Vidanga (Embelia)"

    top_indices = np.argsort(probabilities)[::-1][:3]
    predictions = []
    for index in top_indices:
        key = classes[index] if index < len(classes) else str(index)
        info = _info(info_map, key, f"Class {key}")
        predictions.append({
            "class": key,
            "name": info["name"],
            "description": info["description"],
            "uses": info["uses"],
            "confidence": round(float(probabilities[index]) * 100, 2),
        })

    return {
        "image": display_image,
        "plant_type": plant_type,
        "plant_label": plant_label,
        "plant_confidence": round(plant_confidence, 2),
        "top_prediction": predictions[0],
        "alternatives": predictions[1:],
        "low_confidence": predictions[0]["confidence"] < CONFIDENCE_THRESHOLD,
    }
