import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.fft import fft2, fftshift, ifftshift

# ---------- tus funciones (copiadas tal cual) ----------
rng = np.random.default_rng()
N = 160
offset = 40
sigma = 10.0

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

def random_phase_mask(N, rng):
    return np.exp(1j * 2 * np.pi * rng.random((N, N)))

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

# ---------- datos ----------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
img_left = x_train[0]   # 28x28

# número de imágenes de salida (derecha)
n_out = 5

# ---------- layout (ajustado) ----------
fig = plt.figure(figsize=(12,6))

# eje izquierdo (imagen pequeña) - movido más a la izquierda para reducir gap
ax_left = fig.add_axes([0.12, 0.40, 0.10, 0.12], zorder=5)   # left, bottom, width, height
ax_left.imshow(img_left, cmap="gray", interpolation='nearest', origin='upper')
ax_left.axis("off")
ax_left.set_aspect('equal')

# columna derecha: menos separación horizontal (left_col reducido), ancho ajustado
left_col = 0.4
width = 0.20

top_margin = 0.92
bottom_margin = 0.08
height = 0.22
gap = 0.04

total_needed = n_out * height + (n_out - 1) * gap
avail = top_margin - bottom_margin
if total_needed > avail:
    height = (avail - (n_out - 1) * gap) / n_out
    if height <= 0:
        raise RuntimeError("No hay espacio: reduce n_out o cambia la figura.")

ys_bottoms = []
current_top = top_margin
for i in range(n_out):
    bottom = current_top - height
    ys_bottoms.append(bottom)
    current_top = bottom - gap

ax_rights = []
for bottom in ys_bottoms:
    ax = fig.add_axes([left_col, bottom, width, height], zorder=5)
    jps = compute_jps(img_left, random_phase_mask(N, rng), random_phase_mask(N, rng), N=N, offset=offset, sigma=sigma)
    ax.imshow(jps, cmap="gray", interpolation='nearest', origin='upper')
    ax.axis("off")
    ax.set_aspect('equal')
    ax_rights.append(ax)

# eje para flechas (zorder bajo para quedar DETRÁS de las imágenes)
ax_arrows = fig.add_axes([0, 0, 1, 1], zorder=0)
ax_arrows.set_xlim(0, 1)
ax_arrows.set_ylim(0, 1)
ax_arrows.axis("off")

# calcular posiciones reales de los ejes (en fracciones de figura)
fig.canvas.draw()
left_bbox = ax_left.get_position()
start_x = left_bbox.x0 + left_bbox.width     # borde derecho del ax_left
start_y = left_bbox.y0 + left_bbox.height/2  # centro vertical del ax_left

# dibujar flechas que terminan EN EL BORDE IZQUIERDO de cada imagen derecha (evita tapar)
for ax in ax_rights:
    bb = ax.get_position()
    end_x = bb.x0 - 0.005                      # un poquito antes del borde izquierdo
    end_y = bb.y0 + bb.height/2
    ax_arrows.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops=dict(arrowstyle="->", lw=1.6),
        xycoords='figure fraction',
        textcoords='figure fraction'
    )

# puntos suspensivos visibles: usando fig.text en coordenadas de figura, por debajo de la última imagen derecha
last_bb = ax_rights[-1].get_position()
ellipsis_x = last_bb.x0 + last_bb.width/2
ellipsis_y = last_bb.y0 - 0.04
fig.text(ellipsis_x, ellipsis_y, "...", ha="center", va="top", fontsize=22, zorder=10)

plt.show()



import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# rutas
path_jps =  "C:\Proyectos\Prueba-red\Red_objetounico\input1.png"
path_pred = "C:\Proyectos\Prueba-red\Red_objetounico\output1.png"

# carga
img_jps = mpimg.imread(path_jps)
img_mnist = x_train[2]
img_pred = mpimg.imread(path_pred)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].imshow(img_jps, cmap = 'gray')
axes[0].set_title("JPS")
axes[0].axis("off")

axes[1].imshow(img_mnist, cmap = 'gray')
axes[1].set_title("MNIST")
axes[1].axis("off")

axes[2].imshow(img_pred, cmap = 'gray')
axes[2].set_title("predicción")
axes[2].axis("off")

plt.tight_layout()
plt.show()