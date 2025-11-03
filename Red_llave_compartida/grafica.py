import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- Configuración general ---
n_rows = 1
n_cols = 3
types = ['C:\Proyectos\Prueba-red\Red_llave_compartida\mnist', 'C:\Proyectos\Prueba-red\Red_llave_compartida\input', 'C:\Proyectos\Prueba-red\Red_llave_compartida\output']

fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 15))  # 3 columnas, 5 filas
fig.subplots_adjust(wspace=0.05, hspace=0.1)  # espacio entre imágenes
titles = ["MNIST", "JPS", "PREDICCIÓN"]

for i in range(n_rows):
    for j, t in enumerate(types):
        filename = f"{t}{i+1}.png"  # ej. mnist0.png, input0.png, output0.png
        try:
            img = mpimg.imread(filename)
            ax = axes[ j]
            ax.imshow(img, cmap='gray')
            ax.axis('off')

            if i == 0:  # títulos solo en la primera fila
                ax.set_title(titles[j].upper(), fontsize=12)
        except FileNotFoundError:
            axes[i, j].text(0.5, 0.5, 'No image', ha='center', va='center')
            axes[i, j].axis('off')

plt.tight_layout()
plt.show()