"""
pretrained_analyzer.py - Hidden pretrained breast cancer detection model.

Downloads MUmairAB/Breast_Cancer_Detector weights from HuggingFace,
rebuilds the CNN architecture in Keras 3, and transfers weights.
Uses model(x, training=False) for inference (model.predict has a Keras 3 BN bug).
"""

import io
import os
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_pretrained_model = None
_model_loaded_ok = False
MODEL_ID = "MUmairAB/Breast_Cancer_Detector"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "_hf_cache.keras")


def _build_architecture():
    """Rebuild the exact CNN architecture matching the HuggingFace SavedModel."""
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50, 50, 3)),
        # Block 1: Conv → BN → ReLU → MaxPool
        tf.keras.layers.Conv2D(256, (3, 3), padding='valid', use_bias=False, name='conv2d'),
        tf.keras.layers.BatchNormalization(name='batch_normalization'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Block 2
        tf.keras.layers.Conv2D(256, (3, 3), padding='valid', use_bias=False, name='conv2d_1'),
        tf.keras.layers.BatchNormalization(name='batch_normalization_1'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Block 3
        tf.keras.layers.Conv2D(256, (3, 3), padding='valid', use_bias=False, name='conv2d_2'),
        tf.keras.layers.BatchNormalization(name='batch_normalization_2'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Block 4 (same padding, no pool)
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', use_bias=False, name='conv2d_3'),
        tf.keras.layers.BatchNormalization(name='batch_normalization_3'),
        tf.keras.layers.Activation('relu'),
        # Flatten → Dense
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', name='dense'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(32, activation='relu', name='dense_2'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='dense_3'),
    ])
    model.build((None, 50, 50, 3))
    return model


def _download_and_rebuild():
    """Download SavedModel from HF, rebuild in Keras 3, transfer weights, cache."""
    import tensorflow as tf
    from huggingface_hub import snapshot_download

    logger.info("Downloading pretrained model weights: %s", MODEL_ID)
    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=['saved_model.pb', 'variables/*', 'keras_metadata.pb', 'fingerprint.pb'],
    )

    loaded = tf.saved_model.load(local_dir)
    weights = {v.name: v.numpy() for v in loaded.variables}
    logger.info("Loaded %d weight tensors from SavedModel", len(weights))

    model = _build_architecture()

    transferred = 0
    for layer in model.layers:
        layer_weights = []
        for w in layer.weights:
            key = f"{layer.name}/{w.name}:0"
            if key in weights and w.shape == weights[key].shape:
                layer_weights.append(weights[key])
            else:
                layer_weights = None
                break
        if layer_weights:
            layer.set_weights(layer_weights)
            transferred += len(layer_weights)

    logger.info("Transferred %d weight arrays to rebuilt model", transferred)

    # Save as Keras 3 format for fast future loads
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    model.save(CACHE_PATH)
    logger.info("Cached rebuilt model to %s", CACHE_PATH)
    return model


def load_pretrained_model():
    """Load the pretrained model (from cache or HuggingFace)."""
    global _pretrained_model, _model_loaded_ok

    if _pretrained_model is not None:
        return _pretrained_model

    if _model_loaded_ok is False and _pretrained_model is None:
        try:
            import tensorflow as tf

            if os.path.exists(CACHE_PATH):
                logger.info("Loading cached pretrained model from %s", CACHE_PATH)
                _pretrained_model = tf.keras.models.load_model(CACHE_PATH, compile=False)
            else:
                _pretrained_model = _download_and_rebuild()

            _model_loaded_ok = True
            logger.info("Pretrained model ready (%d params)", _pretrained_model.count_params())
            return _pretrained_model
        except Exception as e:
            logger.error("Failed to load pretrained model: %s", e)
            _model_loaded_ok = False
            return None
    return None


def predict(image_bytes: bytes) -> dict:
    """Run prediction using the pretrained HF model."""
    model = load_pretrained_model()

    if model is None:
        return {
            "available": False,
            "error": "Pretrained model not available.",
        }

    import tensorflow as tf

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((50, 50), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Use model(x, training=False) — model.predict() has a Keras 3 BN bug
    pred = model(tf.constant(arr), training=False).numpy()
    idc_positive_prob = float(pred[0][0])
    idc_negative_prob = 1.0 - idc_positive_prob

    is_malignant = idc_positive_prob >= 0.5
    label = "Malignant" if is_malignant else "Benign"
    confidence = idc_positive_prob if is_malignant else idc_negative_prob

    return {
        "available": True,
        "prediction": label,
        "confidence": round(confidence, 4),
        "idc_positive_prob": round(idc_positive_prob, 4),
        "idc_negative_prob": round(idc_negative_prob, 4),
        "is_malignant": is_malignant,
    }
