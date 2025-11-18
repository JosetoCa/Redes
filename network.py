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
tfrecord_train_pattern = "data_jps_jtcgeneral/train_jps_*.tfrecord"
tfrecord_test_pattern  = "data_jps_jtcgeneral/test_jps_*.tfrecord"
compression = None   # None o "GZIP"  -> debe coincidir con lo usado al escribir
batch_size = 8
shuffle_buffer = 4096
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 4
VIS_SAMPLES = 6      # cuántos ejemplos visualizar del test

# ---------- parser ----------
def parse_tfrecord_normalized(example_proto):
    features = {
        'N': tf.io.FixedLenFeature([], tf.int64),
        'img_idx': tf.io.FixedLenFeature([], tf.int64),
        'key_id': tf.io.FixedLenFeature([], tf.int64),
        'mnist_raw': tf.io.FixedLenFeature([], tf.string),
        'jps_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'jps_png': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    N = tf.cast(parsed['N'], tf.int32)

    def decode_png():
        j = tf.io.decode_png(parsed['jps_png'], channels=1)
        return tf.cast(j, tf.float32) / 255.0

    def decode_raw():
        j = tf.io.decode_raw(parsed['jps_raw'], out_type=tf.uint8)
        j = tf.reshape(j, (N, N, 1))
        return tf.cast(j, tf.float32) / 255.0

    jps = tf.cond(tf.greater(tf.strings.length(parsed['jps_png']), 0), decode_png, decode_raw)
    jps.set_shape([None, None, 1])

    mnist = tf.io.decode_raw(parsed['mnist_raw'], out_type=tf.uint8)
    mnist = tf.reshape(mnist, (28, 28, 1))
    mnist = tf.cast(mnist, tf.float32) / 255.0
    mnist.set_shape([28,28,1])

    return jps, mnist
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
    ds = ds.map(parse_tfrecord_normalized, num_parallel_calls=AUTOTUNE)
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

def build_model_with_skips(input_shape, l2=1e-6, dropout_rate=0.0):
    """
    U-Net style autoencoder (encoder-decoder con skip connections).
    input_shape: (N, N, 1)
    l2: kernel regularization
    dropout_rate: SpatialDropout en bottleneck (opcional)
    """
    def conv_block(x, filters, k=3):
        x = layers.Conv2D(filters, k, padding='same',
                          kernel_regularizer=regularizers.l2(l2),
                          use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    inp = layers.Input(shape=input_shape)   # (N,N,1)

    # --- Encoder (guardar skips) ---
    c1 = conv_block(inp, 32)
    c1 = conv_block(c1, 32)
    p1 = layers.MaxPool2D(2, padding='same')(c1)   # /2

    c2 = conv_block(p1, 64)
    c2 = conv_block(c2, 64)
    p2 = layers.MaxPool2D(2, padding='same')(c2)   # /4

    c3 = conv_block(p2, 128)
    c3 = conv_block(c3, 128)
    p3 = layers.MaxPool2D(2, padding='same')(c3)   # /8  (opcional)

    # Bottleneck
    b = conv_block(p3, 256)
    b = conv_block(b, 256)
    if dropout_rate and dropout_rate > 0:
        b = layers.SpatialDropout2D(dropout_rate)(b)

    # --- Decoder con skips (mirar c3->u3, c2->u2, c1->u1) ---
    u3 = layers.UpSampling2D(2)(b)           # ahora /4
    u3 = layers.Concatenate()([u3, c3])
    u3 = conv_block(u3, 128)
    u3 = conv_block(u3, 128)

    u2 = layers.UpSampling2D(2)(u3)          # ahora /2
    u2 = layers.Concatenate()([u2, c2])
    u2 = conv_block(u2, 64)
    u2 = conv_block(u2, 64)

    u1 = layers.UpSampling2D(2)(u2)          # ahora original tamaño (N,N)
    u1 = layers.Concatenate()([u1, c1])
    u1 = conv_block(u1, 32)
    u1 = conv_block(u1, 32)

    # Forzar salida 28x28 (si el input N != 28) y salida final
    x = layers.Resizing(28, 28, interpolation='bilinear')(u1)
    out = layers.Conv2D(1, 3, padding='same', activation='sigmoid', dtype='float32')(x)

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
    model = build_model_with_skips(input_shape, l2=1e-6, dropout_rate=0.0)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.MeanAbsoluteError()])

    # callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("model_checkpoint.keras", save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss', restore_best_weights=True)
    ]

    # Entrenamiento: ds_train está en repeat=True, por se pasa steps_per_epoch
    train_files = len(glob.glob(tfrecord_train_pattern))
    if train_files == 0:
        raise RuntimeError("No training TFRecord files found. Comprueba el patrón.")
    
    steps_per_epoch = 300000 // batch_size

    # comprueba que la salida esperada (target) y la salida del modelo tengan shape compatible
    # aquí asumimos target (28,28,1)
    assert y_batch.shape[1:] == (28,28,1), f"Target shape inesperada: {y_batch.shape[1:]}"
    # 1. shapes / summary
    model.summary()

    # 2. rango de datos en un batch de test (sanity check)
    for x_batch, y_batch in ds_test.take(1):
        print("jps shape, min/max:", x_batch.shape, x_batch.numpy().min(), x_batch.numpy().max())
        print("target shape, min/max:", y_batch.shape, y_batch.numpy().min(), y_batch.numpy().max())
        break

    # 3. prueba predict antes de entrenar (comprobar que no da error shape/memory)
    x_sample, _ = next(iter(ds_test))
    pred = model.predict(x_sample[:1])
    print("pred shape, min/max:", pred.shape, pred.min(), pred.max())
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