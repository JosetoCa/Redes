"""
Precomputa muchos JPS's para una solo imagen MNIST y guarda como TFRecord(s) usando
distintas llaves de encriptación.
Genera pm_obj.npy y pm_key.npy (llaves) y escribe train/test TFRecords.

Ajustes importantes:
 - N: pixeles del JPS . Con 8GB recomienda 128-160.
 - shards: número de archivos TFRecord para escribir (la idea es usar lectura paralela).
"""
# ---------- librería ----------
# Para el manejo de los archivos
import os
from pathlib import Path
# Para los cálculos y grabado de datos
import numpy as np
import tensorflow as tf
from numpy.fft import fft2, fftshift, ifftshift
from tqdm import tqdm

# ---------- parámetros ----------
N = 160                   # tamaño del JPS 
offset = 40               # separación (pixeles)
sigma = 10.0              # sigma de la gaussiana para key_amp
rng = np.random.default_rng() # Para cambiar la semilla cada vez
out_dir = "data_jps_singleobj" # Carpeta con datos
train_tfrecord_prefix = "train_jps" # Para identificar los roles de los datos encriptados
test_tfrecord_prefix  = "test_jps"
shards = 8                # número de shards para train/test

os.makedirs(out_dir, exist_ok=True)

# ---------- utilidades ----------
def embed_center(container_shape, small):
    """
    Embedes a square window-matrix in the center of a plane-matrix of zeros

    Parameters
    ----------
    container_shape : tuple.
        shape of the biggest matrix.
    small : npdarray NxN
        Square window-matrix.

    Returns
    -------
    out : npdarray container_shape
        Window-matrix embedded in Plane-matrix.
    """
    H, W = container_shape
    h, w = small.shape
    r0 = (H - h) // 2
    c0 = (W - w) // 2
    out = np.zeros((H, W), dtype=small.dtype)
    r1 = max(r0, 0); c1 = max(c0, 0)
    r2 = min(r0 + h, H); c2 = min(c0 + w, W)
    sr0 = r1 - r0; sc0 = c1 - c0
    sr1 = sr0 + (r2 - r1); sc1 = sc0 + (c2 - c1)
    out[r1:r2, c1:c2] = small[sr0:sr1, sc0:sc1]
    return out

def random_phase_mask(N, rng):
    """
    Generates a random phase mask with uniform distibution.

    Parameters
    ----------
    N : int
        Size of the matrix.
    seed : integer
        Random seed for reproducibility.   
 
    Returns
    -------
     : complex ndarray NxN
        random phase mask with uniform distibution.
    """
    return np.exp(1j * 2 * np.pi * rng.random((N, N)))

def shift_field(field, dx=0, dy=0):
    """
    Shifts the entries of a matrix dx and dy by pixels on the horizontal and vertical axis respectively.

    Parameters
    ----------
    field : ndarray NxN
        Matrix to shift.
    dx : integer
        pixel shifted in x-axis.
    dy : integer
        pixel shifted in y-axis.

    Returns
    -------
    : dtype = field
        Shifted matrix.
    """
    return np.roll(np.roll(field, dy, axis=0), dx, axis=1)

def compute_jps(img, pm_obj, pm_key, N=160, offset=40, sigma=10.0):
    """
    computes Joint Power Spectrum (JPS) of a MNIST image given both random phase masks.

    Parameters
    ----------
    img : ndarray 28x28
        Matrix to compute (JPS) from MNIST.
    pm_obj : ndarray NxN
        random phase mask for img.
    pm_key : ndarray NxN
        random phase mask key.
    N : integer
        Matrix shape to compute FFT.
    offset : integer
        pixels between objtect and mask
    sigma : float
        sigma parameter for gauss beam twist at mask

    Returns
    -------
    JPS : ndarray NxN
        JPS of the image given both random phase masks.
    """
        
    # Se embebe la imagen y se multiplica por la máscara de fase
    obj_amp = embed_center((N, N), img)
    obj_field = obj_amp * pm_obj

    # Se genera la llave con una amplitud gaussiana y máscara de fase.
    x = np.linspace(-N/2, N/2-1, N)
    X, Y = np.meshgrid(x, x)
    key_amp = np.exp(-(X**2 + Y**2) / (2 * (sigma**2)))
    key_field = key_amp * pm_key

    # Se separan la máscara y el objeto del centro para poder dejarlos en una sola matriz.
    obj_shifted = shift_field(obj_field, dx=-offset)
    key_shifted = shift_field(key_field, dx=offset)
    joint_field = obj_shifted + key_shifted

    # Se calcula el JPS y se normaliza
    U = fftshift(fft2(ifftshift(joint_field)))
    JPS = np.abs(U)**2
    JPS = JPS.astype(np.float32)
    JPS = (JPS - JPS.min()) / (JPS.max() - JPS.min() + 1e-12)
    return JPS

# ---------- helpers TFRecord ----------
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def write_sharded_tfrecord(prefix, img, rng, N, offset, sigma, out_dir, n, shards=4):
    """
    Escribe 'shards' archivos TFRecord con prefijo dado en out_dir.
    """
    # Calcula las particiones del conjunto de imágenes.

    counts = [n // shards + (1 if i < (n % shards) else 0) for i in range(shards)]

    start = 0

    for shard_idx, cnt in enumerate(counts):
        # Escribe cada shard
        if cnt == 0:
            continue
        # posición del dato final
        end = start + cnt
        # contruye la ruta al archivo TFRecord con su sufijo correcto.
        fname = os.path.join(out_dir, f"{prefix}_{shard_idx:03d}.tfrecord")

        # Para escribir los datos en el disco.
        writer = tf.io.TFRecordWriter(fname)
        print(f"Writing {fname}  (samples {start}:{end})")
        for i in tqdm(range(start, end), desc=f"shard {shard_idx}", unit="sample"):
            # Calcula el JPS de la imagen correspondiente
            jps = compute_jps(img, random_phase_mask(N, rng), random_phase_mask(N, rng), N=N, offset=offset, sigma=sigma)
            # Crea el diccionario que se va a escribir el TFRecord
            feat = {
                'jps_raw': _bytes_feature(jps.tobytes()),
                'mnist_raw'  : _bytes_feature(img.astype(np.float32).tobytes()),
                'N'      : _int64_feature(int(N))
            }
            # Crea el objeto que reconoce TFRecords
            ex = tf.train.Example(features=tf.train.Features(feature=feat))
            # Serializa (SerializeToString) el Example a binario y lo escribe en el archivo .tfrecord abierto.
            writer.write(ex.SerializeToString())
        writer.close()
        start = end

# ---------- main ----------
# Para ejecutar como archivo directamente, no como módulo.
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # preparar la imagen
    img = x_train[0].astype(np.float32)
    if img.max() > 1.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    # metadata
    meta = dict(N=int(N), offset=int(offset), sigma=float(sigma),  use_log=False)
    np.save(os.path.join(out_dir, "metadata.npy"), meta)
    print("Saved keys and metadata in", out_dir)

    # escribir TFRecords shardeados
    write_sharded_tfrecord(train_tfrecord_prefix, img, rng, N, offset, sigma, out_dir, n=50000, shards=shards)
    write_sharded_tfrecord(test_tfrecord_prefix,  img,  rng, N, offset, sigma, out_dir, n=10000, shards=max(1, shards//4))

    print("Done.")
