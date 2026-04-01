"""High-accuracy transfer learning training for IDC histopathology classification."""

import argparse
import json
import os
import random
import subprocess
import sys
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3, EfficientNetB7, DenseNet121

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)
SEED = 42
BASE_DIR = Path(__file__).parent
SAVE_DIR = BASE_DIR / "saved_model"
SAVE_DIR.mkdir(exist_ok=True)
MODEL_PATH = str(SAVE_DIR / "breast_cancer_model.keras")
DATASET_PATH = BASE_DIR / "dataset"
REPORT_PATH = SAVE_DIR / "metrics_report.json"
CM_PATH = SAVE_DIR / "confusion_matrix.png"

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class BinaryFocalLoss(tf.keras.losses.Loss):
    """Focal loss for stronger focus on difficult samples."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, name: str = "binary_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        focal_weight = alpha_factor * tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(focal_weight * bce)


def configure_runtime():
    """Enable accelerators and mixed precision when beneficial."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("GPU detected. Mixed precision enabled.")
        except Exception as exc:
            logger.warning("Could not enable mixed precision: %s", exc)
    else:
        logger.info("No GPU detected. Running on CPU.")


def download_dataset(dataset_root: Path | None = None) -> Path:
    """Resolve dataset location, then try kagglehub and kaggle CLI fallback."""
    if dataset_root is not None and dataset_root.exists():
        return dataset_root

    candidates = [
        DATASET_PATH,
        Path.home() / ".cache/kagglehub/datasets/paultimothymooney/breast-histopathology-images/versions/1",
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info("Using existing dataset at: %s", candidate)
            return candidate

    try:
        import kagglehub

        logger.info("Downloading IDC dataset via kagglehub...")
        path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")
        resolved = Path(path)
        logger.info("Dataset downloaded to: %s", resolved)
        return resolved
    except Exception as exc:
        logger.warning("kagglehub download failed: %s", exc)

    logger.info("Trying Kaggle CLI fallback...")
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "paultimothymooney/breast-histopathology-images",
        "-p",
        str(DATASET_PATH),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
        return DATASET_PATH
    except Exception as exc:
        logger.error("Kaggle CLI fallback failed: %s", exc)
        logger.error("Set up Kaggle credentials and retry.")
        logger.error("Manual fallback: download from Kaggle and extract into %s", DATASET_PATH)
        sys.exit(1)


def collect_image_paths(dataset_root: Path):
    """Collect all PNG paths and labels from the IDC folder structure."""
    benign_paths, malignant_paths = [], []
    for img_path in dataset_root.rglob("*.png"):
        parent = img_path.parent.name
        if parent == "0":
            benign_paths.append(str(img_path))
        elif parent == "1":
            malignant_paths.append(str(img_path))

    logger.info("Found %s benign and %s malignant images", f"{len(benign_paths):,}", f"{len(malignant_paths):,}")
    if not benign_paths or not malignant_paths:
        logger.error("No valid images found under %s", dataset_root)
        sys.exit(1)
    return benign_paths, malignant_paths


def load_and_preprocess(path: str, label: int):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.float32)


def augment(image, label):
    """Stronger geometric and color augmentation for robust generalization."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.25)
    image = tf.image.random_contrast(image, lower=0.75, upper=1.30)
    image = tf.image.random_saturation(image, lower=0.75, upper=1.30)
    image = tf.image.random_hue(image, max_delta=0.04)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def mixup_batch(images, labels, alpha: float = 0.2):
    """MixUp regularization on batched tensors."""
    batch_size = tf.shape(images)[0]
    lam = tfp_sample_beta(alpha, alpha, batch_size)
    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])

    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam_x * images + (1.0 - lam_x) * tf.gather(images, indices)
    labels = tf.reshape(labels, [batch_size, 1])
    mixed_labels = lam_y * labels + (1.0 - lam_y) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


def tfp_sample_beta(alpha: float, beta: float, size):
    """Sample beta distribution without tensorflow-probability dependency."""
    gamma_1 = tf.random.gamma([size], alpha)
    gamma_2 = tf.random.gamma([size], beta)
    return gamma_1 / (gamma_1 + gamma_2)


def make_dataset(paths, labels, batch_size: int, augment_data=False, enable_mixup=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if augment_data:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment_data and enable_mixup:
        ds = ds.map(lambda x, y: mixup_batch(x, y, alpha=0.2), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _get_backbone(backbone_name: str):
    normalized = backbone_name.strip().lower()
    if normalized == "efficientnetb0":
        return EfficientNetB0, "EfficientNetB0"
    if normalized == "efficientnetb3":
        return EfficientNetB3, "EfficientNetB3"
    if normalized == "efficientnetb7":
        return EfficientNetB7, "EfficientNetB7"
    if normalized == "densenet121":
        return DenseNet121, "DenseNet121"
    raise ValueError(f"Unsupported backbone '{backbone_name}'. Use EfficientNetB0, EfficientNetB3, EfficientNetB7, or DenseNet121.")


def build_model(backbone_name="DenseNet121", unfreeze_top_n=0, dropout_rate=0.35):
    """Build transfer-learning model (EfficientNet B0/B3/B7 or DenseNet121)."""
    backbone_cls, backbone_label = _get_backbone(backbone_name)
    base = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = True
    if unfreeze_top_n <= 0:
        for layer in base.layers:
            layer.trainable = False
    else:
        for layer in base.layers[:-unfreeze_top_n]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=unfreeze_top_n > 0)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    if backbone_label == "EfficientNetB7":
        hidden_dim = 512
    elif backbone_label == "DenseNet121":
        hidden_dim = 256
    else:
        hidden_dim = 320
    x = layers.Dense(hidden_dim, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return models.Model(inputs, outputs, name=f"BreastCancerDetector_{backbone_label}")


def compute_class_weights(labels_array):
    n_total = len(labels_array)
    n_pos = int(np.sum(labels_array))
    n_neg = n_total - n_pos
    w_neg = n_total / (2.0 * max(n_neg, 1))
    w_pos = n_total / (2.0 * max(n_pos, 1))
    logger.info("Class weights - Benign: %.3f, Malignant: %.3f", w_neg, w_pos)
    return {0: w_neg, 1: w_pos}


def compile_model(model, learning_rate: float, use_focal: bool):
    loss = BinaryFocalLoss(gamma=2.0, alpha=0.35) if use_focal else tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02)
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def build_callbacks(phase_name: str, patience: int):
    return [
        callbacks.EarlyStopping(monitor="val_auc", patience=patience, restore_best_weights=True, mode="max"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(2, patience // 2), min_lr=1e-7),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", save_best_only=True, mode="max"),
        callbacks.TensorBoard(log_dir=str(BASE_DIR / f"logs/{phase_name}")),
        callbacks.CSVLogger(str(BASE_DIR / f"logs/{phase_name}_history.csv")),
    ]


def evaluate_best_threshold(model, val_ds):
    """Find validation threshold maximizing F1."""
    y_true, y_prob = [], []
    for x_batch, y_batch in val_ds:
        probs = model.predict(x_batch, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(tf.reshape(y_batch, [-1]).numpy().tolist())

    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)

    if len(thresholds) == 0:
        return 0.5
    idx = int(np.nanargmax(f1_scores[:-1]))
    best_threshold = float(thresholds[idx])
    logger.info("Best threshold on val set (F1): %.4f", best_threshold)
    return best_threshold


def evaluate_test_set(model, test_ds, threshold: float):
    y_true, y_prob = [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(tf.reshape(y_batch, [-1]).numpy().tolist())

    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int32)

    report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=160)
    plt.close()

    summary = {
        "threshold": threshold,
        "f1": float(f1),
        "accuracy": float(report["accuracy"]),
        "benign_precision": float(report["Benign"]["precision"]),
        "benign_recall": float(report["Benign"]["recall"]),
        "malignant_precision": float(report["Malignant"]["precision"]),
        "malignant_recall": float(report["Malignant"]["recall"]),
        "support": int(len(y_true)),
    }
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved metrics report to: %s", REPORT_PATH)
    logger.info("Saved confusion matrix to: %s", CM_PATH)
    return summary


def train(args):
    configure_runtime()
    logger.info("=" * 68)
    logger.info("Breast Cancer Detection - EfficientNetB7 High-Accuracy Training")
    logger.info("=" * 68)

    dataset_root = download_dataset(Path(args.dataset) if args.dataset else None)
    benign_paths, malignant_paths = collect_image_paths(dataset_root)

    all_paths = benign_paths + malignant_paths
    all_labels = [0] * len(benign_paths) + [1] * len(malignant_paths)

    if args.data_fraction < 1.0:
        all_paths, _, all_labels, _ = train_test_split(
            all_paths,
            all_labels,
            train_size=args.data_fraction,
            stratify=all_labels,
            random_state=SEED,
        )
        logger.info("Using %.0f%% of dataset for faster training: %s samples", args.data_fraction * 100.0, f"{len(all_paths):,}")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=0.2,
        stratify=all_labels,
        random_state=SEED,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=SEED,
    )

    logger.info("Train: %s | Val: %s | Test: %s", f"{len(train_paths):,}", f"{len(val_paths):,}", f"{len(test_paths):,}")

    class_weights = compute_class_weights(np.array(train_labels))

    train_ds = make_dataset(
        train_paths,
        train_labels,
        batch_size=args.batch_size,
        augment_data=True,
        enable_mixup=args.enable_mixup,
    )
    val_ds = make_dataset(val_paths, val_labels, batch_size=args.batch_size, augment_data=False)
    test_ds = make_dataset(test_paths, test_labels, batch_size=args.batch_size, augment_data=False)

    model = build_model(backbone_name=args.backbone, unfreeze_top_n=0, dropout_rate=0.35)
    compile_model(model, learning_rate=1e-3, use_focal=False)
    logger.info("Phase 1/3: Frozen backbone head training")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=class_weights,
        callbacks=build_callbacks("phase1_head", patience=4),
    )

    model = build_model(backbone_name=args.backbone, unfreeze_top_n=args.unfreeze_phase2, dropout_rate=0.35)
    model.load_weights(MODEL_PATH)
    compile_model(model, learning_rate=2e-4, use_focal=True)
    logger.info("Phase 2/3: Fine-tune top %s layers with focal loss", args.unfreeze_phase2)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune_1,
        class_weight=class_weights,
        callbacks=build_callbacks("phase2_finetune60", patience=5),
    )

    model = build_model(backbone_name=args.backbone, unfreeze_top_n=args.unfreeze_phase3, dropout_rate=0.30)
    model.load_weights(MODEL_PATH)
    compile_model(model, learning_rate=7e-5, use_focal=True)
    logger.info("Phase 3/3: Deep fine-tune top %s layers", args.unfreeze_phase3)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune_2,
        class_weight=class_weights,
        callbacks=build_callbacks("phase3_finetune140", patience=6),
    )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    compile_model(model, learning_rate=1e-5, use_focal=False)
    eval_values = model.evaluate(test_ds, verbose=1)
    for metric_name, metric_value in zip(model.metrics_names, eval_values):
        logger.info("Test %s: %.4f", metric_name, metric_value)

    best_threshold = evaluate_best_threshold(model, val_ds)
    summary = evaluate_test_set(model, test_ds, threshold=best_threshold)

    # Save a copy with the backbone-specific name for model.py to find
    backbone_short = args.backbone.replace("EfficientNet", "")
    if args.backbone.lower() == "densenet121":
        backbone_short = "DenseNet121"
    named_path = str(SAVE_DIR / f"model_{backbone_short}.keras")
    model.save(named_path)
    logger.info("Saved named model to: %s", named_path)

    # Also save the optimal threshold
    summary["optimal_threshold"] = best_threshold
    threshold_path = SAVE_DIR / "threshold.json"
    threshold_path.write_text(json.dumps({"threshold": best_threshold}, indent=2), encoding="utf-8")
    logger.info("Saved optimal threshold (%.4f) to: %s", best_threshold, threshold_path)

    logger.info("Best-threshold test F1: %.4f", summary["f1"])
    logger.info("Best-threshold test accuracy: %.4f", summary["accuracy"])
    logger.info("Model saved to: %s", MODEL_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNetB7 for breast cancer detection")
    parser.add_argument("--dataset", type=str, default="", help="Path to extracted IDC dataset root")
    parser.add_argument("--backbone", type=str, default="DenseNet121", choices=["EfficientNetB0", "EfficientNetB3", "EfficientNetB7", "DenseNet121"])
    parser.add_argument("--data-fraction", type=float, default=0.5, help="Fraction of dataset to use (0.1 to 1.0)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-finetune-1", type=int, default=10)
    parser.add_argument("--epochs-finetune-2", type=int, default=10)
    parser.add_argument("--unfreeze-phase2", type=int, default=40)
    parser.add_argument("--unfreeze-phase3", type=int, default=80)
    parser.add_argument("--enable-mixup", action="store_true", help="Enable mixup regularization (slower)")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
