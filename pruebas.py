import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Clase_propagacion as prop
import Clase_objetosEntrada as OE
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image


# 1. Cargar MNIST (ya preparado por Keras)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Para simular la encriptación con el JTC debemos definir 2 funciones
# dos máscaras de fase, una que multiplica el objeto, otro que se pone
# en el espectro de Fourier.
def embed_at(container_shape, small, topleft):
    H, W = container_shape
    h, w = small.shape
    out = np.zeros((H, W), dtype=small.dtype)

    r0, c0 = topleft
    r1 = max(r0, 0)
    c1 = max(c0, 0)
    r2 = min(r0 + h, H)
    c2 = min(c0 + w, W)

    sr0 = r1 - r0
    sc0 = c1 - c0
    sr1 = sr0 + (r2 - r1)
    sc1 = sc0 + (c2 - c1)

    out[r1:r2, c1:c2] = small[sr0:sr1, sc0:sc1]
    return out
def embed_center(container_shape, small):
    H, W = container_shape
    h, w = small.shape
    r0 = (H - h) // 2
    c0 = (W - w) // 2
    return embed_at(container_shape, small, (r0, c0))
def random_phase_mask(N, seed=None):
    rng = np.random.default_rng(seed)
    return np.exp(1j * 2 * np.pi * rng.random((N, N)))


def shift_field(field, dx=0, dy=0):
    """Traslada un campo en el espacio real usando desplazamiento entero (wrap-around)."""
    return np.roll(np.roll(field, dy, axis=0), dx, axis=1)


# ----------------------- parámetros -----------------------
N = 160                    # tamaño matriz
offset = 40                # desplazamiento horizontal (en píxeles) entre objeto y llave
seed = 1234                 # semilla reproducible
path = "C:\Proyectos\Prueba-red\letra.png"

# crear objeto y llave
im = Image.open(path).convert("L")  # grayscale
arr = np.array(im).astype(np.float32)
obj_amp = embed_center((N,N), x_train[0])
pm_obj = random_phase_mask(N, seed=seed)
pm_key = random_phase_mask(N, seed=9999)

# campo del objeto y de la llave (amplitud * fase aleatoria)
obj_field = obj_amp * pm_obj/np.max(obj_amp)
# llave: para el ejemplo usamos una distribución gaussiana como amplitud modulada por fase
x = np.linspace(-N/2, N/2-1, N)
X, Y = np.meshgrid(x, x)
sigma = 10
key_amp = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
key_field = key_amp * pm_key

# desplazar objeto a la izquierda y llave a la derecha (simula posiciones separadas en el plano de entrada)
obj_shifted = shift_field(obj_field, dx=-offset)
key_shifted = shift_field(key_field, dx=offset)

# campo conjunto de entrada (suma)
joint_field = obj_shifted + key_shifted

# ----------------------- encriptación (Joint Power Spectrum) -----------------------
U = fftshift(fft2(ifftshift(joint_field)))      # transformada de Fourier del campo conjunto
JPS = np.abs(U)**2                               # Joint Power Spectrum (intensidad)

# ----------------------- intento de desencriptado simple -----------------------
# calculamos la FT de la llave por separado (la posición de la llave debe coincidir con la usada en la encriptación)
K = fftshift(fft2(ifftshift(key_shifted)))

# multiplicamos el JPS por la conjugada de la FT de la llave y aplicamos IFT
# Esta operación intenta extraer el término cruzado A B* que contiene la información del objeto.
G = JPS * np.conj(K)
recon_complex = fftshift(ifft2(ifftshift(G)))
recon = np.abs(recon_complex)

# normalizar para visualización
recon = recon / (recon.max() + 1e-12)

# ----------------------- visualización -----------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
ax = axes.ravel()

ax[0].imshow(np.abs(joint_field)**2, cmap='gray')
ax[0].set_title('Objeto (amplitud)')
ax[0].axis('off')

ax[1].imshow(np.angle(pm_obj), cmap='twilight')
ax[1].set_title('Fase aleatoria (obj)')
ax[1].axis('off')

ax[2].imshow(np.angle(pm_key), cmap='twilight')
ax[2].set_title('Fase aleatoria (key)')
ax[2].axis('off')

ax[3].imshow(np.log10(JPS + 1e-12), cmap='inferno')
ax[3].set_title('Joint Power Spectrum (log)')
ax[3].axis('off')

ax[4].imshow(JPS, cmap='gray')
ax[4].set_title('JPS (intensidad)')
ax[4].axis('off')

ax[5].imshow(recon, cmap='gray')
ax[5].set_title('Reconstrucción tentativa (abs)')
ax[5].axis('off')

plt.tight_layout()
plt.imsave("C:\Proyectos\Prueba-red\input_letra.png", JPS/np.max(JPS), cmap="gray") 


plt.show()

# ----------------------- comentarios finales -----------------------
print("Hecho. La reconstrucción mostrada es un ejemplo didáctico. Para mejorarla se suele: ")
print(" - aplicar filtrado en el plano de Fourier para seleccionar solo uno de los términos cruzados")
print(" - usar ventanas o máscaras de fase conocidas como claves (keys) con mayor complejidad")
print(" - trabajar con datos experimentales y calibración de desplazamientos y fases.")


#-----------------------------------------------------------------------


# ----------------------- parámetros -----------------------
N = 1024                    # tamaño matriz
offset = 100                # desplazamiento horizontal (en píxeles) entre objeto y llave
seed = 42                  # semilla reproducible

# crear objeto y llave
obj_amp = embed_center((N,N), x_train[0])
pm_obj = random_phase_mask(N, seed=seed)
pm_key = random_phase_mask(N, seed=seed+1)

# campo del objeto y de la llave (amplitud * fase aleatoria)
obj_field = obj_amp * pm_obj/np.max(obj_amp)
# llave: para el ejemplo usamos una distribución gaussiana como amplitud modulada por fase
x = np.linspace(-N/2, N/2-1, N)
X, Y = np.meshgrid(x, x)
sigma = 20
key_amp = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
key_field = key_amp * pm_key

# desplazar objeto a la izquierda y llave a la derecha (simula posiciones separadas en el plano de entrada)
obj_shifted = shift_field(obj_field, dx=-offset)
key_shifted = shift_field(key_field, dx=offset)

# campo conjunto de entrada (suma)
joint_field = obj_shifted + key_shifted

# ----------------------- encriptación (Joint Power Spectrum) -----------------------
f = 200
p = prop.propa(L = 20, N=N, z=f)
lente = OE.objPlanoEntrada(L=15,N=N).centerThinLens(f=f,wl = p.wl)
U = p.propa(p.propa(joint_field)*lente)      # transformada de Fourier del campo conjunto
JPS = np.abs(U)**2                               # Joint Power Spectrum (intensidad)

# ----------------------- intento de desencriptado simple -----------------------
# calculamos la FT de la llave por separado (la posición de la llave debe coincidir con la usada en la encriptación)
K = p.propa(p.propa(key_shifted)*lente)
# multiplicamos el JPS por la conjugada de la FT de la llave y aplicamos IFT
# Esta operación intenta extraer el término cruzado A B* que contiene la información del objeto.
G = JPS * np.conj(K)
recon_complex = p.propa(p.propa(G)*lente)
recon = np.abs(recon_complex)

# normalizar para visualización
recon = recon / (recon.max() + 1e-12)

# ----------------------- visualización -----------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
ax = axes.ravel()

ax[0].imshow(np.abs(joint_field)**2, cmap='gray')
ax[0].set_title('Objeto (amplitud)')
ax[0].axis('off')

ax[1].imshow(np.angle(pm_obj), cmap='twilight')
ax[1].set_title('Fase aleatoria (obj)')
ax[1].axis('off')

ax[2].imshow(np.angle(pm_key), cmap='twilight')
ax[2].set_title('Fase aleatoria (key)')
ax[2].axis('off')

ax[3].imshow(np.log10(JPS + 1e-12), cmap='inferno')
ax[3].set_title('Joint Power Spectrum (log)')
ax[3].axis('off')

ax[4].imshow(JPS, cmap='gray')
ax[4].set_title('JPS (intensidad)')
ax[4].axis('off')

ax[5].imshow(recon, cmap='gray')
ax[5].set_title('Reconstrucción tentativa (abs)')
ax[5].axis('off')

plt.tight_layout()
plt.show()
