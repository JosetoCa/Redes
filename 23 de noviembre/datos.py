"""
Precomputa 100 JPS's para cada imagen MNIST y guarda como TFRecord(s) usando
distintas 100 llaves de encriptación distintas.
Genera key_seeds.npy (semillas para reproducir las llaves), metadata.npy 
y escribe train/test TFRecords.
"""
# ---------- librería ----------
# Para el manejo de los archivos
import os
import json
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
out_dir = "data_jps_jtcgeneral" # Carpeta con datos
train_tfrecord_prefix = "train_jps" # Para identificar los roles de los datos encriptados
test_tfrecord_prefix  = "test_jps"
shards = 50                # número de shards para train/test

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

def random_phase_mask(seed, N):
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
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(N, N)).astype(np.float32)
    return np.exp(1j * phases)  # máscara compleja con amplitud 1

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

def save_global_keys(out_dir, num_keys=100, master_seed=123456):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(int(master_seed))
    key_seeds = rng.integers(0, 2**31-1, size=int(num_keys), dtype=np.int64)
    np.save(os.path.join(out_dir, "key_seeds.npy"), key_seeds)
    meta = {"num_keys": int(num_keys), "master_seed": int(master_seed)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print("Saved:", os.path.join(out_dir, "key_seeds.npy"), "and meta.json")

# ---------- helpers TFRecord ----------
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def write_dataset_with_global_keys(mnist_images, out_dir, key_seeds_path,
                                   N=160, shards=256, use_png=False, prefix="data"):
    """
    Ahora escribe archivos con nombre: {prefix}_{shard_idx:04d}.tfrecord
    mnist_images: numpy array shape (num_images,28,28)
    key_seeds_path: path a key_seeds.npy
    prefix: string
    """
    os.makedirs(out_dir, exist_ok=True)
    key_seeds = np.load(key_seeds_path)
    num_keys = len(key_seeds)
    num_images = int(mnist_images.shape[0])
    total_samples = num_images * num_keys

    counts = [ total_samples // shards + (1 if i < total_samples % shards else 0)
               for i in range(shards) ]

    sample_idx = 0
    for shard_idx, cnt in enumerate(counts):
        if cnt == 0:
            continue
        tfname = os.path.join(out_dir, f"{prefix}_{shard_idx:04d}.tfrecord")
        writer = tf.io.TFRecordWriter(tfname)
        for _ in tqdm(range(cnt), desc=f"{prefix} shard {shard_idx:04d}"):
            img_idx = sample_idx // num_keys
            key_id = sample_idx % num_keys

            img = mnist_images[img_idx]
            if img.dtype != np.float32:
                img_f = img.astype(np.float32) / 255.0
            else:
                img_f = np.clip(img, 0.0, 1.0)
            if img_f.ndim == 3 and img_f.shape[-1] == 1:
                img_f = img_f.squeeze(-1)

            pm_obj = random_phase_mask(seed=img_idx + 100000, N=N)
            pm_key = random_phase_mask(seed=int(key_seeds[key_id]), N=N)

            jps = compute_jps(img_f, pm_obj, pm_key, N=N, offset=offset, sigma=sigma)

            jps_u8 = (np.clip(jps, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            mnist_u8 = (np.clip(img_f, 0.0, 1.0) * 255.0).round().astype(np.uint8)

            features = {
                'img_idx': _int64_feature(img_idx),
                'key_id' : _int64_feature(key_id),
                'N'      : _int64_feature(N),
            }

            if use_png:
                jps_tf = tf.convert_to_tensor(jps_u8, dtype=tf.uint8)
                if jps_tf.ndim == 2:
                    jps_tf = tf.expand_dims(jps_tf, -1)
                png_bytes = tf.io.encode_png(jps_tf).numpy()
                features['jps_png'] = _bytes_feature(png_bytes)
            else:
                features['jps_raw'] = _bytes_feature(jps_u8.tobytes())

            features['mnist_raw'] = _bytes_feature(mnist_u8.tobytes())

            ex = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(ex.SerializeToString())

            sample_idx += 1
        writer.close()
        print("Wrote shard:", tfname)
    print("Done. total samples:", sample_idx)
# ---------- main ----------
# Para ejecutar como archivo directamente, no como módulo.
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    meta = dict(N=int(N), offset=int(offset), sigma=float(sigma))
    np.save(os.path.join(out_dir, "metadata.npy"), meta)
    x_train = x_train[:len(x_train)//2] # A la mitad son 25000
    x_test = x_test[:len(x_test)//2] # A la mitad son 5000

    # guarda las N keys globales (misma para train y test)
    save_global_keys(out_dir, num_keys=100, master_seed=20251123)
    key_seeds_path = os.path.join(out_dir, "key_seeds.npy")

    # escribe train con prefijo explícito
    print("Escribiendo train...")
    write_dataset_with_global_keys(x_train, out_dir, key_seeds_path, N=N, shards=shards, use_png=False, prefix=train_tfrecord_prefix)

    # escribe test con prefijo distinto
    print("Escribiendo test...")
    write_dataset_with_global_keys(x_test, out_dir, key_seeds_path, N=N, shards=max(1, shards//4), use_png=False, prefix=test_tfrecord_prefix)

    print("Terminado. key_seeds en:", key_seeds_path)