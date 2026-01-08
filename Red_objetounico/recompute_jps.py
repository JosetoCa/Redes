# recompute_jps.py
import os
import numpy as np
import tensorflow as tf
from numpy.fft import fft2, fftshift, ifftshift

# ------------------ utilidades (copiadas/adaptadas de tu generador) ------------------
def embed_center(container_shape, small):
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
    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(N, N)).astype(np.float32)
    return np.exp(1j * phases)

def shift_field(field, dx=0, dy=0):
    return np.roll(np.roll(field, dy, axis=0), dx, axis=1)

def compute_jps(img, pm_obj, pm_key, N=160, offset=40, sigma=10.0):
    obj_amp = embed_center((N, N), img)
    obj_field = obj_amp * pm_obj

    x = np.linspace(-N/2, N/2-1, N)
    X, Y = np.meshgrid(x, x)
    key_amp = np.exp(-(X**2 + Y**2) / (2 * (sigma**2)))
    key_field = key_amp * pm_key

    obj_shifted = shift_field(obj_field, dx=-offset)
    key_shifted = shift_field(key_field, dx=offset)
    joint_field = obj_shifted + key_shifted

    U = fftshift(fft2(ifftshift(joint_field)))
    JPS = np.abs(U)**2
    JPS = JPS.astype(np.float32)
    JPS = (JPS - JPS.min()) / (JPS.max() - JPS.min() + 1e-12)
    return JPS

# ------------------ funciones de ayuda I/O ------------------
def load_npy(path):
    arr = np.load(path, allow_pickle=True)
    print(f"Loaded {path} -> shape: {getattr(arr,'shape',None)}, dtype: {arr.dtype}")
    return arr

def read_tfrecord_get_sample(tfrecord_path, sample_index=0):
    """
    Lee un TFRecord (formato usado por tu script) y devuelve un dict con:
      - 'img_idx', 'key_id', 'N', 'mnist' (28x28 float), 'jps' (N x N uint8 si existe)
    """
    feature_desc = {
        'img_idx': tf.io.FixedLenFeature([], tf.int64),
        'key_id' : tf.io.FixedLenFeature([], tf.int64),
        'N'      : tf.io.FixedLenFeature([], tf.int64),
        # jps_raw or jps_png may be present:
        'jps_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'jps_png': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'mnist_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }

    ds = tf.data.TFRecordDataset([tfrecord_path])
    ds = ds.skip(sample_index).take(1)
    for raw in ds:
        ex = tf.io.parse_single_example(raw, feature_desc)
        img_idx = int(ex['img_idx'].numpy())
        key_id = int(ex['key_id'].numpy())
        N = int(ex['N'].numpy())

        result = {'img_idx': img_idx, 'key_id': key_id, 'N': N}

        if ex['mnist_raw'].numpy():
            mnist_bytes = ex['mnist_raw'].numpy()
            mnist_u8 = np.frombuffer(mnist_bytes, dtype=np.uint8)
            # MNIST stored as 28x28 (in generator it was flattened bytes)
            # try to infer shape:
            if mnist_u8.size == 28*28:
                mnist = mnist_u8.reshape((28,28)).astype(np.float32) / 255.0
            else:
                # if it's 28*28*1:
                mnist = mnist_u8.reshape((28,28,1)).squeeze(-1).astype(np.float32) / 255.0
            result['mnist'] = mnist

        if ex['jps_raw'].numpy():
            jps_bytes = ex['jps_raw'].numpy()
            jps_u8 = np.frombuffer(jps_bytes, dtype=np.uint8)
            jps = jps_u8.reshape((N, N))
            result['jps'] = jps
        elif ex['jps_png'].numpy():
            # Si está en PNG, podemos guardarlo en disco o decodificar con tf.io.decode_png
            png_bytes = ex['jps_png'].numpy()
            img_tf = tf.io.decode_png(png_bytes)
            jps = img_tf.numpy().squeeze(-1)
            result['jps'] = jps

        return result
    raise ValueError("No sample found in TFRecord.")

# ------------------ generación de llave personalizada ------------------
def get_pm_key_from_seed_or_array(seed_or_array, N):
    """
    seed_or_array can be:
     - int/np.integer -> generates random_phase_mask(seed, N)
     - numpy array (N,N) with complex phase mask -> returned (validated)
    """
    if isinstance(seed_or_array, (int, np.integer)):
        return random_phase_mask(seed_or_array, N)
    elif isinstance(seed_or_array, np.ndarray):
        if seed_or_array.shape != (N, N):
            raise ValueError(f"Array shape must be {(N,N)}")
        return seed_or_array.astype(np.complex64)
    else:
        raise TypeError("seed_or_array must be int or np.ndarray")

# ------------------ ejemplo de uso ------------------
if __name__ == "__main__":
    
    # 1) Abrir key_seeds.npy y metadata.npy (si existen en la carpeta por defecto)
    data_dir = "C:\Proyectos\Prueba-red\data_jps_jtcgeneral"  # mismo nombre que en tu script
    key_seeds_path = os.path.join(data_dir, "key_seeds.npy")
    metadata_path = os.path.join(data_dir, "metadata.npy")

    if os.path.exists(key_seeds_path):
        key_seeds = load_npy(key_seeds_path)
    else:
        print("No se encontró key_seeds.npy en", data_dir)
        key_seeds = None

    if os.path.exists(metadata_path):
        meta = load_npy(metadata_path).item() if metadata_path.endswith(".npy") else None
        print("metadata:", meta)
    else:
        print("No se encontró metadata.npy en", data_dir)
        meta = None

    

    # 3) Recalcular un JPS usando una llave que SEPAS que es distinta.
    #    -> Opción A: usas un seed que no esté en key_seeds.npy (recomendado)
    #    -> Opción B: cargas un .npy con la máscara compleja (N x N)


    # Ejemplo A: elegir seed personalizado y comprobar que no está en key_seeds
    custom_seed = 99999999
    if key_seeds is not None:
        if np.any(key_seeds == custom_seed):
            print(f"ADVERTENCIA: custom_seed {custom_seed} está en key_seeds.npy. Selecciona otro.")
        else:
            print(f"custom_seed {custom_seed} NO está en key_seeds.npy -> OK para usarlo.")
    print(key_seeds)