"""
Network.py — pipeline y entrenamiento para: entrada = JPS, salida = imagen MNIST.
Lectura paralela de shards, parser compatible con los TFRecords.
Incluye visualización de resultados sobre ejemplos de test.
"""
import glob
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- GPU setup ---
gpus = tf.config.list_physical_devices('GPU') # Lista las GPU detectadas por TF
if gpus:
    try:
        for gpu in gpus:
            # Habilita el crecimiento dinámico de memoria en cada GPU.
            # Asigna VRAM según se vaya usando.
            tf.config.experimental.set_memory_growth(gpu, True) 
        print(f"Usando GPU: {gpus}")
    except Exception as e:
        print("Error al configurar GPU:", e)

# --- desactivar mixed precision por compatibilidad/estabilidad ---
# Para lograr compatabilidad entre los cálculos en GPU y CPU se fija una política
# global para las variables, todas en float32
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('float32')   # usar float32 en todo
    print("Usando float32 (mixed precision desactivada) para evitar " \
    "conflictos XLA/mixed-precision.")
except Exception as e:
    print("No se pudo configurar mixed precision (se sigue con float32):", e)


# --- fix: desactivar XLA para evitar conflicto con mixed precision ---
# XLA es un compilador de álgebra lineal que acelera los modelos de TF sin 
# cambiar el código fuente, lo desactivo porque no he sabido implementarlo con
# mixed_precision, necesaria para el XLA.
tf.config.optimizer.set_jit(False)
print("XLA desactivado (evita conflicto mixed precision + ReluGrad)")


# ---------- parámetros ----------
# Esto propio de la forma en que se crean los .tfrecord en datos.py
tfrecord_train_pattern = "data_jps_jtcgeneral/train_jps_*.tfrecord"
tfrecord_test_pattern  = "data_jps_jtcgeneral/test_jps_*.tfrecord"
compression = None   # None o "GZIP"  -> debe coincidir con lo usado al escribir
# Cantidad de ejemplos sobre los que se calcula el gradiente para 
# actualizar pesos y bias.
batch_size = 50
# el shuffle_buffer es la cantidad de muestras del dataset que viven en RAM sobre 
# las que se hacen permutaciones para evitar que la red encuentre patrones
# por el orden del dataset.
shuffle_buffer = 4096
# El Autotune hace que TF establezca cuántos elementos procesar en paralelo en el ds.map
# en prefetech para que automáticamente ponga los batches en la GPU
AUTOTUNE = tf.data.AUTOTUNE
# Estos procesos de paralelización están automatizados en TF y termina siendo sencillo
# implementarlos en este entorno.
EPOCHS =20
VIS_SAMPLES = 6      # cuántos ejemplos visualizar del test

# ---------- parser ----------
# Esta es la función que toma un ejemplo del tfrecord y  devuelve el JPS y el MNIST
def parse_tfrecord_normalized(example_proto):
    # Estas son las posibles características en un ejemplo del .tfrecord
    features = {
        'N': tf.io.FixedLenFeature([], tf.int64),
        'img_idx': tf.io.FixedLenFeature([], tf.int64),
        'key_id': tf.io.FixedLenFeature([], tf.int64),
        'mnist_raw': tf.io.FixedLenFeature([], tf.string),
        'jps_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'jps_png': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    # Este es el elemento parseado, example_proto es el ejemplo que se obtiene
    # apartir del reader de los datasets (ver make_dataset)
    parsed = tf.io.parse_single_example(example_proto, features)
    # Se extrea la característica "N" y se establece en .int32.
    # Extraerla implica tomar los bits relacionados a la feature 'N' y darles un formato
    N = tf.cast(parsed['N'], tf.int32)
    # Toma los bits codificados PNG, los decodifica y da el formato float32 y normaliza.
    def decode_png():
        j = tf.io.decode_png(parsed['jps_png'], channels=1)
        return tf.cast(j, tf.float32) / 255.0
    # Toma los bits del feature 'jps_raw' y les da un formato float32 y normaliza.
    def decode_raw():
        j = tf.io.decode_raw(parsed['jps_raw'], out_type=tf.uint8)
        j = tf.reshape(j, (N, N, 1))
        return tf.cast(j, tf.float32) / 255.0
    # Se elige entre decodificar PNG o RAW, según lo que haya en los features.
    jps = tf.cond(tf.greater(tf.strings.length(parsed['jps_png']), 0), decode_png, decode_raw)
    jps.set_shape([None, None, 1])
    # Toma los bits RAW del MNIST y lo transforma a formato tf y normaliza
    mnist = tf.io.decode_raw(parsed['mnist_raw'], out_type=tf.uint8)
    mnist = tf.reshape(mnist, (28, 28, 1))
    mnist = tf.cast(mnist, tf.float32) / 255.0
    mnist.set_shape([28,28,1])

    return jps, mnist
# ---------- dataset (lectura paralela de shards) ----------
def make_dataset(tfrecord_pattern, batch_size, shuffle=True, compression=None, repeat=False):
    # Se construye un dataset de elementos 'strings' que son las rutas a los .tfrecords
    files = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=shuffle)
    # El reader crea un Dataset (objeto para leer el .tfrecord) para leer los ejemplos
    # contenidos en los .tfrecord
    # se pasa buffer_size grande al TFRecordDataset para mejorar read throughput
    def reader(f):
        return tf.data.TFRecordDataset(f, compression_type=compression, buffer_size=8*1024*1024)
    # interleave abre múltiples datasets haciendo uso del reader
    # mezcla los elementos provenientes los distintos datasets y produce un un
    # cycle_length es el número de datasets abiertos y num_parallel_calls la cantidad
    # de llamadas del dataset en paralelo.
    ds = files.interleave(
        lambda f: reader(f),
        cycle_length=AUTOTUNE, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    # El shuffle se encarga de combinar (revolver) los ejemplos llamados para mayor azar.
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    # El mapa toma los bits que da el reader y parsea los datos para que estén listos a
    # entrenar la red, (JPS, MNIST). Este parseo se hace en paralelo con el AUTOTUNE.
    ds = ds.map(parse_tfrecord_normalized, num_parallel_calls=AUTOTUNE)
    # repeat se usa para que 
    if repeat:
        ds = ds.repeat()
    # Agrupa las muestras parseadas por el map en batches
    ds = ds.batch(batch_size)
    # si hay GPU, hace prefetch (pone los datos inmediatamente en la GPU) para cuando se
    # necesiten.
    if tf.config.list_physical_devices('GPU'):
        ds = ds.apply(tf.data.experimental.prefetch_to_device('/GPU:0'))
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------- modelo: U-net simple (entrada: JPS -> salida: 28x28 image) ----------
from tensorflow.keras import layers, models, regularizers

def build_model_with_skips(input_shape, l2=1e-6, dropout_rate=0.0):
    """
    U-Net style autoencoder (encoder-decoder con skip connections).
    input_shape: (N, N, 1)
    l2: kernel regularization
    dropout_rate: SpatialDropout en bottleneck (opcional)
    """
    # Bloque convolucional:   
    # Capa convolucional, con padding = 'same' para que se preserve el tamño
    #   y sea más fácil conectar encoder y decoder. Con regularización l2, para evitar 
    #   overfitting. Sin bais, pues el Batch_normalitazion agrega un factor y un sumando.
    # Capa de Batch_normalitazion, normaliza alrededor de 0 y asigna una desviación 
    #   estándar.
    # Capa de activación a la que entra el valor lineal de las capas anteriores. 
    #   Se usa Relu.  
    def conv_block(x, filters, k=3):
        x = layers.Conv2D(filters, k, padding='same',
                          kernel_regularizer=regularizers.l2(l2),
                          use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    # La entrada, el JPS
    inp = layers.Input(shape=input_shape)   # (N,N,1)

    # --- Encoder (guardar skips) ---
    # Se pone MaxPool con kernel = 2 y padding='same', lo que reduce a la 
    # mitad la resolución de la entrada.
    c1 = conv_block(inp, 32)
    c1 = conv_block(c1, 32)
    p1 = layers.MaxPool2D(2, padding='same')(c1)   # /2

    c2 = conv_block(p1, 64)
    c2 = conv_block(c2, 64)
    p2 = layers.MaxPool2D(2, padding='same')(c2)   # /4

    c3 = conv_block(p2, 128)
    c3 = conv_block(c3, 128)
    p3 = layers.MaxPool2D(2, padding='same')(c3)   # /8 

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

    # En esta capa se hace una interpolación por promedio para reducir la 
    # la resolución de la imagen de salida. La U-net es por definición simétrica,
    # si se quiere que la salida sea un MNIST, se vuelve necesario o romper la
    # simetría o aplicar o resizing.
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

    # crear los datasets de entrenamiento y validación
    ds_train = make_dataset(tfrecord_train_pattern, batch_size=batch_size, shuffle=True, compression=compression, repeat=True)
    ds_test  = make_dataset(tfrecord_test_pattern,  batch_size=batch_size, shuffle=False, compression=compression, repeat=False)
    # Toma 1 batch de entremiento para su shape, tipo y rango
    for x,y in ds_train.take(1):
        print("train batch shapes:", x.shape, y.shape, "dtype:", x.dtype, y.dtype, "range y:", tf.reduce_min(y).numpy(), tf.reduce_max(y).numpy())
    # obtener input shape del primer batch
    for x_batch, y_batch in ds_train.take(1):
        input_shape = x_batch.shape[1:]   # (N,N,1)
    model = build_model_with_skips(input_shape, l2=1e-6, dropout_rate=0.0)
    # El optimizador del modelo se establece con Adam y un LR de 10^{-4}
    # La función de costo es la entropía cruzada binaria y se agrega
    # como métrica el valor absoluto medio, para saber qué tan cerca va de la MNIST,
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.MeanAbsoluteError()])

    # callbacks
    # Se guarda el mejor modelo según el valor de la función de pérdida.
    # Se reduce el LR si val_loss no mejora. También utiliza EarlyStopping.
    callbacks = [
    tf.keras.callbacks.ModelCheckpoint("model_checkpoint.keras", save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss', restore_best_weights=True),
    tf.keras.callbacks.CSVLogger("training_log.csv", append=False)   # guarda history en CSV
    ]

    # Entrenamiento: ds_train está en repeat=True, por se pasa steps_per_epoch
    train_files = len(glob.glob(tfrecord_train_pattern)) # Se cuentan los shards.
    # De este modo se recorre todo el dataset
    steps_per_epoch = 3000000 // batch_size 

    # comprueba que la salida esperada y la salida del modelo tengan shape compatible
    # aquí asumimos target (28,28,1)
    assert y_batch.shape[1:] == (28,28,1), f"Target shape inesperada: {y_batch.shape[1:]}"
    
    # shapes / summary
    model.summary() # imprime el número de capas y parámetros de la red

    # prueba predict antes de entrenar (comprobar que no da error shape/memory)
    # toma un batch del dataset de test, luego predicepara el primer ejemplo del batch
    # esto para verificar que el modelo construido funciona
    x_sample, _ = next(iter(ds_test))
    pred = model.predict(x_sample[:1])
    print("pred shape, min/max:", pred.shape, pred.min(), pred.max())

    # Aquí está el entremaniendo de la red
    history = model.fit(
    ds_train,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=ds_test,
    callbacks=callbacks
    )

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