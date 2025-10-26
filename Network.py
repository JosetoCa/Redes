"""
Network.py — pipeline y entrenamiento para: entrada = JPS, salida = imagen MNIST.
Lectura paralela de shards, parser compatible con los TFRecords actuales
(donde el campo de la "etiqueta" es 'mnist_raw' con la imagen 28x28 guardada como float32).
Incluye visualización de resultados sobre ejemplos de test.
"""
import glob
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- GPU setup ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Usando GPU: {gpus}")
    except Exception as e:
        print("Error al configurar GPU:", e)

# --- desactivar mixed precision por compatibilidad/estabilidad ---
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('float32')   # usar float32 en todo
    print("Usando float32 (mixed precision desactivada) para evitar conflictos XLA/mixed-precision.")
except Exception as e:
    print("No se pudo configurar mixed precision (se sigue con float32):", e)


# --- fix: desactivar XLA para evitar conflicto con mixed precision ---
tf.config.optimizer.set_jit(False)
print("XLA desactivado (evita conflicto mixed precision + ReluGrad)")


# ---------- parámetros ----------
tfrecord_train_pattern = "data_jps_singlekey/train_jps_*.tfrecord"
tfrecord_test_pattern  = "data_jps_singlekey/test_jps_*.tfrecord"
compression = None   # None o "GZIP"  -> debe coincidir con lo usado al escribir
batch_size = 8
shuffle_buffer = 4096
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 20
VIS_SAMPLES = 6      # cuántos ejemplos visualizar del test

# ---------- parser ----------
def parse_tfrecord(example_proto):
    """
    Parse a single TFRecord example and return (jps, target_image).
    jps: shape (N, N, 1), float32, nominalmente ya en [0,1]
    target: shape (28, 28, 1), float32, normalizada a [0,1]
    """
    features = {
        'jps_raw'  : tf.io.FixedLenFeature([], tf.string),
        'mnist_raw': tf.io.FixedLenFeature([], tf.string),
        'N'        : tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, features)

    # N puede ser tensor; construimos reshape dinámico y luego fijamos shape estática
    N = tf.cast(parsed['N'], tf.int32)

    # decode JPS
    jps = tf.io.decode_raw(parsed['jps_raw'], tf.float32)
    jps = tf.reshape(jps, tf.stack([N, N]))
    jps = tf.expand_dims(jps, -1)               # (N, N, 1)
    # Hacer shape estática si se conoce N en tiempo de diseño:
    jps.set_shape([None, None, 1])              # deja que TFGraph lo determine si no es fijo

    # decode MNIST target
    target = tf.io.decode_raw(parsed['mnist_raw'], tf.float32)   # guardaste float32
    target = tf.reshape(target, (28, 28))
    target = tf.expand_dims(target, -1)         # (28, 28, 1)
    # Normalizamos target a [0,1] — en tu script guardaste valores 0..255 como float32
    target = target / 255.0
    target.set_shape([28, 28, 1])

    return jps, target

# ---------- dataset (lectura paralela de shards) ----------
def make_dataset(tfrecord_pattern, batch_size=64, shuffle=True, compression=None, repeat=False):
    files = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    # se pasa buffer_size grande al TFRecordDataset para mejorar read throughput
    def reader(f):
        return tf.data.TFRecordDataset(f, compression_type=compression, buffer_size=8*1024*1024)
    ds = files.interleave(
        lambda f: reader(f),
        cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    # si hay GPU, prefetch directo a dispositivo para quitar overhead de host->device
    if tf.config.list_physical_devices('GPU'):
        ds = ds.apply(tf.data.experimental.prefetch_to_device('/GPU:0'))
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------- modelo: autoencoder simple (entrada: JPS -> salida: 28x28 image) ----------
from tensorflow.keras import layers, models, regularizers

def build_model(input_shape,
                l2=1e-4,
                dropout_rate=0.15):
    """
    Autoencoder con BatchNorm + L2 + SpatialDropout en bottleneck.
    Parámetros:
      - l2: factor de regularización L2 para kernels (kernel_regularizer)
      - dropout_rate: SpatialDropout2D en bottleneck (0 = sin dropout)
    """
    # helper para evitar repetición
    def conv_block(x, filters, k=3, stride=1, name=None):
        x = layers.Conv2D(filters, k, padding='same',
                          kernel_regularizer=regularizers.l2(l2),
                          use_bias=False)(x)   # bias omitido por BatchNorm
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    inp = layers.Input(shape=input_shape)

    # Encoder
    x = conv_block(inp, 25)
    x = layers.MaxPool2D(2)(x)

    x = conv_block(x, 50)
    x = layers.MaxPool2D(2)(x)

    x = conv_block(x, 128)
    # Bottleneck
    x = conv_block(x, 128)
    if dropout_rate and dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)

    # Decoder
    x = layers.UpSampling2D(2)(x)
    # si querés usar skip connections: concatená aquí con la salida previa (opcional)
    x = conv_block(x, 64)

    x = layers.UpSampling2D(2)(x)
    x = conv_block(x, 32)
    x = conv_block(x, 16)

    # Forzar salida 28x28 y conv final (sin BatchNorm; salida en float32)
    x = layers.Resizing(28, 28)(x)
    out = layers.Conv2D(1, 3, padding='same',
                        activation='sigmoid',
                        dtype='float32',
                        kernel_regularizer=regularizers.l2(l2))(x)

    model = models.Model(inputs=inp, outputs=out)
    return model


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


# ---------- main ----------
if __name__ == "__main__":

    # datasets
    ds_train = make_dataset(tfrecord_train_pattern, batch_size=batch_size, shuffle=True, compression=compression, repeat=True)
    ds_test  = make_dataset(tfrecord_test_pattern,  batch_size=batch_size, shuffle=False, compression=compression, repeat=False)
    # try:
    #     ds_test = ds_test.cache()
    #     print("ds_test cached")
    # except Exception:
    #     pass
    for x,y in ds_train.take(1):
        print("train batch shapes:", x.shape, y.shape, "dtype:", x.dtype, y.dtype, "range y:", tf.reduce_min(y).numpy(), tf.reduce_max(y).numpy())


    # obtener input shape del primer batch
    for x_batch, y_batch in ds_train.take(1):
        input_shape = x_batch.shape[1:]   # (N,N,1)
    print("Input shape:", input_shape)
    model = build_model(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='mse',                          # reconstrucción: MSE
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    # callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("model_checkpoint.keras", save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss', restore_best_weights=True)
    ]

    # Entrenamiento: ds_train está en repeat=True, por eso pasamos steps_per_epoch
    train_files = len(glob.glob(tfrecord_train_pattern))
    if train_files == 0:
        raise RuntimeError("No training TFRecord files found. Comprueba el patrón.")
    # estimar samples: leemos metadata (opcional) o inferimos tamaño aproximado:
    steps_per_epoch = 60000 // batch_size   # sustitúyelo si no usas MNIST completo

    # comprueba que la salida esperada (target) y la salida del modelo tengan shape compatible
    # aquí asumimos target (28,28,1)
    assert y_batch.shape[1:] == (28,28,1), f"Target shape inesperada: {y_batch.shape[1:]}"
    model.fit(ds_train,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              validation_data=ds_test,
              callbacks=callbacks)

    # --- Visualizar predicciones sobre un pequeño lote de test ---
    # tomamos un batch de ds_test
    for x_batch, y_batch in ds_test.take(1):
        x_np = x_batch.numpy()
        y_np = y_batch.numpy()
        preds = model.predict(x_np)
        preds = preds.astype(np.float32)
        visualize_predictions(x_np, y_np, preds, n=VIS_SAMPLES, save_path="predicciones.png")
        break