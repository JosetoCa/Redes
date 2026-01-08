# inference_fixed.py
import os
import sys
import zipfile
import traceback
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# IMPORTA tu build_robust_model desde autoencoder.py
# Asegúrate de que autoencoder.py esté en el mismo folder o en PYTHONPATH
from autoencoder import build_robust_model

# --- Ajustes básicos ---
MODEL_WEIGHTS = r"C:\\Proyectos\\Prueba-red\\final.weights.h5"
MODEL_FULL = r"C:\\Proyectos\\Prueba-red\\final_model.keras"
INPUT_PATH = r"C:\\Proyectos\\Prueba-red\\input3.png"
OUTPUT_PATH = r"C:\\Proyectos\\Prueba-red\\output3.png"
ASSUME_JPS_RANGE_0_1 = True
TARGET_SIZE = (160, 160)  # tu modelo espera 160x160; si quieres forzar otro tamaño, cámbialo


def print_versions():
    import sys
    print("Python:", sys.version.replace("\n", " "))
    print("TF:", tf.__version__)
    print("Keras backend:", tf.keras.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs visibles:", gpus)


def try_list_keras_zip(path):
    if not os.path.exists(path):
        print("No existe:", path)
        return
    try:
        with zipfile.ZipFile(path, 'r') as z:
            print("Contenido de", path)
            for n in z.namelist():
                print("  ", n)
    except zipfile.BadZipFile:
        print(path, "no es un ZIP válido o está corrupto (o es SavedModel dir).")


def load_model_prefer_weights():
    """Intenta: 1) cargar arquitectura + weights (.h5), 2) fallback: load_model(full)"""
    print_versions()

    # 1) reconstruir arquitectura y cargar pesos si existen
    model = build_robust_model((160, 160, 1))
    if os.path.exists(MODEL_WEIGHTS):
        try:
            model.load_weights(MODEL_WEIGHTS)
            print("Pesos cargados desde:", MODEL_WEIGHTS)
            return model
        except Exception as e:
            print("Error al cargar pesos:", e)
            traceback.print_exc()

    # 2) fallback: intentar cargar el modelo entero (puede fallar por marshal)
    if os.path.exists(MODEL_FULL):
        try:
            print("Intentando tf.keras.models.load_model con unsafe deserialization...")
            tf.keras.config.enable_unsafe_deserialization()
            m = tf.keras.models.load_model(MODEL_FULL, compile=False)
            print("Modelo completo cargado desde:", MODEL_FULL)
            return m
        except Exception as e:
            print("load_model completo falló con excepción:")
            traceback.print_exc()
            # inspeccion rápido del .keras
            try_list_keras_zip(MODEL_FULL)

    raise RuntimeError("No pude cargar el modelo ni los pesos. Revisa rutas y archivos.")


def preprocess_image_from_file(path, target_size=(160,160), assume_range_0_1=True):
    im = Image.open(path).convert("L")
    if target_size is not None:
        im = im.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
    arr = np.array(im).astype(np.float32)
    if assume_range_0_1 and arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)   # H,W,1
    arr = np.expand_dims(arr, axis=0)    # 1,H,W,1
    return arr


def save_prediction(pred_np, out_path):
    # pred_np: (B, H, W, C) o (H,W,C)
    if pred_np.ndim == 4:
        img = pred_np[0, :, :, 0]
    elif pred_np.ndim == 3:
        img = pred_np[:, :, 0]
    else:
        raise ValueError("Forma inesperada pred:", pred_np.shape)
    img = np.clip(img, 0.0, 1.0)
    img255 = (img * 255.0).astype(np.uint8)
    Image.fromarray(img255).save(out_path)
    print("Predicción guardada en:", out_path)


if __name__ == "__main__":
    try:
        model = load_model_prefer_weights()
    except Exception as e:
        print("ERROR FATAL: no se pudo obtener modelo:", e)
        sys.exit(1)

    img_in = preprocess_image_from_file(INPUT_PATH, target_size=TARGET_SIZE, assume_range_0_1=ASSUME_JPS_RANGE_0_1)
    print("Input shape (preprocessed):", img_in.shape, "dtype:", img_in.dtype)

    pred = model.predict(img_in)
    print("Output shape:", pred.shape, "min/max:", pred.min(), pred.max())
    save_prediction(pred, OUTPUT_PATH)
