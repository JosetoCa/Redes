import zipfile, json, os
keras_path  = r"C:\Proyectos\Prueba-red\final_model.keras"


print("Es archivo zip?", zipfile.is_zipfile(keras_path))
with zipfile.ZipFile(keras_path, "r") as z:
    for name in z.namelist():
        print(name)
    # intenta imprimir model config si existe
    for candidate in ("model.json", "model_config.json", "model_config", "keras_metadata.json", "saved_model.pb"):
        if candidate in z.namelist():
            print("\n--- Contenido de", candidate, "---")
            print(z.read(candidate).decode("utf-8")[:4000])