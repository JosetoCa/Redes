# inference.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Ajustes básicos ---
MODEL_PATH = "C:\Proyectos\Prueba-red\Red_llave_compartida\model_checkpoint.keras"   # fichero Keras que guardaste con ModelCheckpoint
INPUT_PATH = "C:\Proyectos\Prueba-red\Red_llave_compartida\input3.png"                 # archivo de entrada (cambiar según necesites)
OUTPUT_PATH = "C:\Proyectos\Prueba-red\Red_llave_compartida\output4.png"             # archivo a escribir con la predicción
ASSUME_JPS_RANGE_0_1 = True              # si tus JPS están en 0..1. Si no, ajusta escala.
TARGET_SIZE = None   # si None: intenta inferir N desde la imagen; si quieres forzar N usa (N,N)

# --- util: cargar modelo ---
def load_trained_model(path):
    print("Cargando modelo desde:", path)
    model = tf.keras.models.load_model(path, compile=False)  # compile no es necesario para inferencia
    print("Modelo cargado. Input shape esperado:", model.input_shape)
    return model

# --- util: preprocesado para imagen desde disco ---
def preprocess_image_from_file(path, target_size=None, assume_range_0_1=True):
    """
    Lee imagen (grayscale) y devuelve tensor (1, N, N, 1), dtype float32.
    Si target_size es (N,N) redimensiona. Si None, usa tamaño original.
    """
    im = Image.open(path).convert("L")  # grayscale
    if target_size is not None:
        im = im.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
    arr = np.array(im).astype(np.float32)
    # Normalizar:
    if assume_range_0_1:
        # si la imagen está en 0..255 -> 0..1
        if arr.max() > 1.0:
            arr = arr / 255.0
    # expand dims a (1, H, W, 1)
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

# --- util: postprocesado y guardado ---
def save_prediction(pred_np, out_path):
    """
    pred_np: array (B, 28, 28, 1) o (28,28,1). Guarda la primera imagen como PNG (0..255).
    """
    if pred_np.ndim == 4:
        img = pred_np[0,...,0]
    elif pred_np.ndim == 3:
        img = pred_np[...,0]
    else:
        raise ValueError("Forma inesperada pred:", pred_np.shape)
    # clamp 0..1
    img = np.clip(img, 0.0, 1.0)
    img255 = (img * 255.0).astype(np.uint8)
    Image.fromarray(img255).save(out_path)
    print("Predicción guardada en:", out_path)

# --- util: visualizar en pantalla (opcional) ---
def show_triplet(jps, target, pred):
    # jps, target, pred deben ser numpy arrays con shape (H,W) o (H,W,1)
    def to2d(a):
        a = np.squeeze(a)
        return a
    jps = to2d(jps); target = to2d(target); pred = to2d(pred)
    plt.figure(figsize=(8,3))
    plt.subplot(1,3,1); plt.imshow(jps, cmap='gray'); plt.title("JPS"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(target, cmap='gray'); plt.title("Target"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(pred, cmap='gray'); plt.title("Pred"); plt.axis('off')
    plt.tight_layout(); plt.show()

# ---------- MAIN (ejemplo: usar imagen desde archivo) ----------
if __name__ == "__main__":
    model = load_trained_model(MODEL_PATH)

    # 1) Si quieres usar una imagen en disco como JPS:
    #    Ajusta TARGET_SIZE si tu modelo espera N x N distinto del que tienes.
    img_in = preprocess_image_from_file(INPUT_PATH, target_size=TARGET_SIZE, assume_range_0_1=ASSUME_JPS_RANGE_0_1)
    print("Input shape (preprocessed):", img_in.shape, "dtype:", img_in.dtype)

    # 2) Inferencia (usa la GPU si TF la detecta)
    pred = model.predict(img_in)
    print("Output shape:", pred.shape, "dtype:", pred.dtype, "min/max:", pred.min(), pred.max())

    # 3) Guardar y mostrar
    save_prediction(pred, OUTPUT_PATH)

    # Si tenés target disponible y querés visualizar:
    # target_arr = ... cargar target equivalente (28x28 normalizado 0..1)
    # show_triplet(img_in[0,...,0], target_arr[...,0], pred[0,...,0])
