import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from pathlib import Path


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
def show_9_mixed_columns(items, col_titles=None, figsize=(9,9), gray_cmap='gray'):
    if len(items) != 9:
        raise ValueError("Se requieren exactamente 9 elementos.")
    if col_titles is None:
        col_titles = ["", "", ""]

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    k = 0
    for col in range(3):
        for row in range(3):
            ax = axes[row, col]
            it = items[k]
            k += 1

            img = None
            if isinstance(it, (str, Path)):
                try:
                    img = mpimg.imread(str(it))
                except:
                    pass
            else:
                try:
                    arr = np.array(it)
                    if arr.ndim >= 2:
                        img = arr
                except:
                    pass

            if img is None:
                ax.imshow(np.ones((100,100,3))*0.85)
                ax.text(0.5,0.5,"no encontrada",ha='center',va='center',transform=ax.transAxes)
            else:
                if img.ndim == 2:
                    ax.imshow(img, cmap=gray_cmap)
                else:
                    ax.imshow(img)
            ax.axis('off')

    # Títulos de columnas
    x_centers = [1/6, 0.5, 5/6]
    for x, t in zip(x_centers, col_titles):
        fig.text(x, 0.95, t, ha='center')
    plt.subplots_adjust(wspace=0.03)
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()
ruta = "C:\Proyectos\Prueba-red"
# EJEMPLO de uso:
if __name__ == "__main__":
    import numpy as np
    arrays = [ruta+"\inputa.png", ruta+"\inputc.png", ruta+"\inputg.png"]
    paths = [ruta+r"\am.png",ruta+"\cosa.png", ruta+"\god.png", 
             ruta+"\outputa.png", ruta+"\outputc.png", ruta+"\outputg.png"]
    mixed = arrays + paths  # total 9 (orden: fila a fila, izquierda->derecha)
    col_titles = ["JPS", "Objeto", "Predicción"]
    show_9_mixed_columns(mixed, col_titles)
