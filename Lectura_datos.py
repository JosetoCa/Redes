"""
Verificador: lee unos ejemplos del TFRecord, reconstruye la JPS a numpy y muestra JPS vs imagen MNIST original.
Útil para comprobar que la JPS coincide con la imagen esperada.
"""
import glob
import tensorflow as tf
import matplotlib.pyplot as plt

tfrecord_pattern = "C:\Proyectos\Prueba-red\data_jps_singlekey/train_jps_*.tfrecord"

files = sorted(glob.glob(tfrecord_pattern))
ds = tf.data.TFRecordDataset(files)
it = iter(ds)
num_show = 5  # cantidad de ejemplos que quieres mostrar
fig, axes = plt.subplots(num_show, 2, figsize=(6, 2*num_show))  # 2 columnas: MNIST | JPS

for i in range(num_show):
    example = next(it)
    parsed = tf.io.parse_single_example(
        example,
        {
            'jps_raw': tf.io.FixedLenFeature([], tf.string),
            'mnist_raw': tf.io.FixedLenFeature([], tf.string),
            'N': tf.io.FixedLenFeature([], tf.int64)
        }
    )

    # Extraer dimensiones y decodificar tensores
    N = int(parsed['N'].numpy())
    jps = tf.io.decode_raw(parsed['jps_raw'], tf.float32)
    jps = tf.reshape(jps, (N, N)).numpy()
    plt.imsave(f"C:\Proyectos\Prueba-red\input{i}.png", jps, cmap="gray") 

    mnist = tf.io.decode_raw(parsed['mnist_raw'], tf.float32)
    mnist = tf.reshape(mnist, (28, 28)).numpy()

    # --- Graficar imagen MNIST ---
    ax_mnist = axes[i, 0] if num_show > 1 else axes[0]
    ax_mnist.imshow(mnist, cmap='gray')
    ax_mnist.set_title(f"MNIST #{i}")
    ax_mnist.axis('off')

    # --- Graficar JPS ---
    ax_jps = axes[i, 1] if num_show > 1 else axes[1]
    ax_jps.imshow(jps, cmap='gray')
    ax_jps.set_title(f"JPS #{i}")
    ax_jps.axis('off')

plt.tight_layout()
plt.show()