"""
Network.py — Arquitectura Robusta (ResNet Autoencoder)
Diseñada para generalización en datasets masivos (3M muestras).
- Sin U-Net Skips (evita paso de ruido).
- Bloques Residuales (alta capacidad de aprendizaje).
- Flujo de datos continuo (sin pausas largas).
"""
import glob
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers

# --- GPU Setup ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)

# Mixed Precision para RTX 4090 (Velocidad x2)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

# ==========================================
# PARÁMETROS DE ENTRENAMIENTO (ESTRATEGIA 3M)
# ==========================================
# No intentamos recorrer los 3M en una sola "epoch" de Keras.
# Hacemos "checkpoints" visuales cada 2000 pasos.
BATCH_SIZE = 128
VIRTUAL_STEPS_PER_EPOCH = 2000  # ~256,000 imágenes por "época" reportada
VIS_SAMPLES = 6
# Si tienes 3M de datos, recorrerás el dataset completo cada ~12 épocas virtuales.
epochs = 50
tfrecord_train_pattern = "data_jps_jtcgeneral/train_jps_*.tfrecord"
tfrecord_test_pattern  = "data_jps_jtcgeneral/test_jps_*.tfrecord"
AUTOTUNE = tf.data.AUTOTUNE

# ==========================================
# PARSER & DATASET
# ==========================================
def parse_tfrecord_optimized(example_proto):
    features = {
        'N': tf.io.FixedLenFeature([], tf.int64),
        'mnist_raw': tf.io.FixedLenFeature([], tf.string),
        'jps_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    
    # Entrada: JPS (160x160)
    jps = tf.io.decode_raw(parsed['jps_raw'], out_type=tf.uint8)
    jps = tf.reshape(jps, (160, 160, 1))
    jps = tf.cast(jps, tf.float32) / 255.0
    
    # Salida: MNIST (28x28)
    mnist = tf.io.decode_raw(parsed['mnist_raw'], out_type=tf.uint8)
    mnist = tf.reshape(mnist, (28, 28, 1))
    mnist = tf.cast(mnist, tf.float32) / 255.0
    
    return jps, mnist

def make_dataset(tfrecord_pattern, batch_size=128, shuffle=True, repeat=True):
    # repeat=True es CRÍTICO para entrenar sobre los 3M sin reiniciar el iterador
    files = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    
    def reader(f):
        return tf.data.TFRecordDataset(f, buffer_size=16*1024*1024) # Buffer grande
    
    ds = files.interleave(reader, cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(10000) # Shuffle buffer decente
    
    ds = ds.map(parse_tfrecord_optimized, num_parallel_calls=AUTOTUNE)
    
    if repeat:
        ds = ds.repeat() 
        
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# ==========================================
# ARQUITECTURA: DEEP RESNET AUTOENCODER
# ==========================================
def residual_block(x, filters, stride=1):
    """Bloque residual estándar para aprendizaje profundo estable"""
    shortcut = x
    # Si cambiamos dimensiones (stride > 1), ajustamos el shortcut
    if stride > 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)

    # Rama principal
    x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Suma + Activación final
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_robust_model(input_shape):
    inp = layers.Input(shape=input_shape)

    # --- ENCODER PROFUNDO ---
    # Extraer características abstractas del JPS
    x = layers.Conv2D(32, 7, strides=2, padding='same')(inp) # 80x80
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 32)            # 80x80
    x = residual_block(x, 64, stride=2)  # 40x40
    x = residual_block(x, 64)            # 40x40
    x = residual_block(x, 128, stride=2) # 20x20
    x = residual_block(x, 128)           # 20x20
    x = residual_block(x, 256, stride=2) # 10x10
    
    # --- BOTTLENECK (La "Magia") ---
    # Aquí la red debe descartar la llave y quedarse solo con el dígito
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x) 
    x = layers.Dropout(0.3)(x) # Ayuda a generalizar ante llaves nuevas
    
    # Proyección al Decoder
    x = layers.Dense(7 * 7 * 256, activation='relu')(x)
    x = layers.Reshape((7, 7, 256))(x)

    # --- DECODER ---
    # Reconstrucción limpia
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x) # 14x14
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)  # 28x28
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Salida final
    out = layers.Conv2D(1, 3, padding='same')(x)
    out = layers.Activation('sigmoid', dtype='float32', name='output_sigmoid')(out)

    return models.Model(inputs=inp, outputs=out)

# ---------- util: visualizar batch de predicciones ----------
def visualize_predictions(jps_batch, target_batch, pred_batch, n=6, save_path=None):
    """
    Entrada: numpy arrays:
      jps_batch: (B, N, N, 1)
      target_batch: (B, 28, 28, 1)
      pred_batch: (B, 28, 28, 1)
    Muestra n ejemplos con columnas: JPS | target | pred
    Si save_path no es None, guarda la figura en ese path (PNG). Si no, hace plt.show().
    """
    n = min(n, jps_batch.shape[0])
    plt.figure(figsize=(12, 4*n))
    for i in range(n):
        jps = jps_batch[i, ..., 0]
        tar = target_batch[i, ..., 0]
        pred = pred_batch[i, ..., 0]

        ax = plt.subplot(n, 3, 3*i + 1)
        ax.imshow(jps, cmap='gray')
        ax.set_title("JPS")
        ax.axis('off')

        ax = plt.subplot(n, 3, 3*i + 2)
        ax.imshow(tar, cmap='gray', vmin=0, vmax=1)
        ax.set_title("Target (MNIST)")
        ax.axis('off')

        ax = plt.subplot(n, 3, 3*i + 3)
        ax.imshow(pred, cmap='gray', vmin=0, vmax=1)
        ax.set_title("Predicción")
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        # crear carpeta si hace falta
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()
        plt.close()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Dataset infinito para train
    ds_train = make_dataset(tfrecord_train_pattern, batch_size=BATCH_SIZE, repeat=True)
    # Dataset finito para validación (usamos una parte para que sea rápido)
    ds_test  = make_dataset(tfrecord_test_pattern, batch_size=BATCH_SIZE, repeat=False).take(100)

    model = build_robust_model((160, 160, 1))
    model.summary()

    # Compilación: AdamW suele generalizar mejor que Adam
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0),
                  loss='binary_crossentropy',
                  metrics=['mae'])
    callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/best.weights.h5",
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1, min_lr = 1e-6),
    tf.keras.callbacks.CSVLogger("training_log.csv")
    ]
    print(f"--- INICIANDO ENTRENAMIENTO ROBUSTO ---")
    print(f"Total datos disponibles: ~3,000,000")
    print(f"Pasos por reporte (época virtual): {VIRTUAL_STEPS_PER_EPOCH}")
    print(f"Esto permite ver progreso cada ~15 min sin detener el flujo de datos.")

    history = model.fit(
        ds_train,
        epochs=epochs, 
        steps_per_epoch=VIRTUAL_STEPS_PER_EPOCH, 
        validation_data=ds_test,
        callbacks=callbacks,
        verbose=1 
    )
    model.save_weights("final.weights.h5")
    model.save("final_model_saved.keras")

   # --- Visualizar predicciones sobre un pequeño lote de test ---
    # tomamos un batch de ds_test
    for x_batch, y_batch in ds_test.take(1):
        x_np = x_batch.numpy()
        y_np = y_batch.numpy()
        preds = model.predict(x_np)
        preds = preds.astype(np.float32)
        visualize_predictions(x_np, y_np, preds, n=VIS_SAMPLES, save_path="predicciones.png")
        break
    hist = history.history  # dict: 'loss','val_loss', 'mean_absolute_error', 'val_mean_absolute_error', ...

    # 1) Grafica Loss (train / val)
    plt.figure(figsize=(8,5))
    plt.plot(hist.get('loss', []), label='train loss')
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss durante entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.close()

    # 2) Graficar todas las métricas (cada métrica en su propia figura)
    # Detecta métricas distintas a loss/val_loss y grafica train/val si existen
    metric_keys = [k for k in hist.keys() if k not in ('loss', 'val_loss')]
    # obtener nombres base (sin 'val_' si aplica)
    base_metrics = sorted(set(k[4:] if k.startswith('val_') else k for k in metric_keys))

    for m in base_metrics:
        train_k = m
        val_k = 'val_' + m
        if train_k in hist:
            plt.figure(figsize=(8,5))
            plt.plot(hist[train_k], label=f'train {m}')
            if val_k in hist:
                plt.plot(hist[val_k], label=f'val {m}')
            plt.xlabel('Epoch')
            plt.ylabel(m)
            plt.title(f'{m} durante entrenamiento')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            fname = f"training_{m}.png".replace('/', '_')
            plt.savefig(fname)
            plt.close()

    # 3) Guardar history completo en NPZ para análisis posterior
    np.savez('training_history.npz', **{k: np.array(v) for k, v in hist.items()})

    print("Guardado: training_loss.png, training_<metric>.png, training_log.csv, training_history.npz")