"""
Generate a small DenseNet121-based model for pipeline testing.
NOTE: This model has RANDOM weights and will NOT give meaningful predictions.
Run train.py to get a real trained model.
"""
import os
import json
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121


def create_dummy_densenet_model():
    """Create a small DenseNet121 transfer-learning model (untrained)."""
    base = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BreastCancerDetector_DenseNet121")
    return model


def generate_and_save(saved_model_dir: str | None = None) -> str:
    """
    Generate a dummy DenseNet121 model and save it to disk.
    Called automatically on first server startup when no trained model exists.
    Returns the path to the saved model file.
    """
    if saved_model_dir is None:
        saved_model_dir = os.path.join(os.path.dirname(__file__), "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)

    print("⚙️  Generating dummy DenseNet121 model (ImageNet backbone + random head)...")
    model = create_dummy_densenet_model()
    model_path = os.path.join(saved_model_dir, "model_DenseNet121.keras")
    model.save(model_path)
    print(f"   Saved: {model_path} ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")
    print(f"   Params: {model.count_params():,}")

    threshold_path = os.path.join(saved_model_dir, "threshold.json")
    if not os.path.exists(threshold_path):
        with open(threshold_path, "w") as f:
            json.dump({"threshold": 0.5}, f, indent=2)

    print("✅ Dummy model ready. Actual predictions use the downloaded analysis model.")
    return model_path


if __name__ == "__main__":
    saved_model_dir = os.path.join(os.path.dirname(__file__), "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)

    print("Creating DenseNet121-based model (ImageNet weights, untrained classifier head)...")
    model = create_dummy_densenet_model()
    model_path = os.path.join(saved_model_dir, "model_DenseNet121.keras")
    model.save(model_path)
    print(f"Saved to {model_path} ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")
    print(f"Total params: {model.count_params():,}")

    # Save default threshold
    threshold_path = os.path.join(saved_model_dir, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"threshold": 0.5}, f, indent=2)
    print(f"Saved default threshold to {threshold_path}")

    print("\n⚠️  WARNING: This model has an untrained classifier head.")
    print("   Predictions will be unreliable. Run train.py for a proper model.")
