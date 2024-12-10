import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Define las rutas de entrada y salida
input_dir = "./archive/split"  # Carpeta raíz con subcarpetas 'train', 'test', y 'val'
output_dir = "./split_processed"  # Carpeta donde se guardarán las imágenes procesadas

# Crear directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_image(image_path):
    # Leer la imagen
    image = cv2.imread(image_path)
    
    # Verificar si la imagen se ha cargado correctamente
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None, None
    
    # Crear una lista para almacenar las imágenes de cada etapa
    stages = []
    
    # Paso 1: Imagen original
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    stages.append(("Original", original_rgb))
    
    # Paso 2: Convertir a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    stages.append(("Convertida a HSV", hsv_rgb))
    
    # Paso 3: Suavizado
    blurred_image = cv2.GaussianBlur(hsv_image, (3, 3), 0)
    blurred_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_HSV2RGB)
    stages.append(("Suavizada", blurred_rgb))
    
    # Paso 4: Separar y procesar canal V
    h, s, v = cv2.split(blurred_image)
    stages.append(("Canal H", h))
    stages.append(("Canal S", s))
    stages.append(("Canal V", v))
    v_equalized = cv2.equalizeHist(v) # se usa para ecualizar el histograma 
    v_contrast = cv2.normalize(v_equalized, None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX) # se usa para normalizar la imagen 
    edges = cv2.Canny(v_contrast, 20, 60)
    v_combined = cv2.addWeighted(v_contrast, 0.9, edges, 0.1, 0)
    stages.append(("Ecualización", v_equalized))
    stages.append(("Contraste en V", v_contrast))
    stages.append(("Bordes", edges))
    stages.append(("Canal V Combinado", v_combined))
    
    
    
    # Paso 5: Recomponer HSV y convertir a BGR
    processed_hsv = cv2.merge([h, s, v_combined])
    processed_bgr = cv2.cvtColor(processed_hsv, cv2.COLOR_HSV2BGR)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    stages.append(("Procesada", processed_rgb))
    
    return processed_bgr, stages

def show_all_stages(stages):
    # Crear un collage con las imágenes de las diferentes etapas
    num_stages = len(stages)
    fig, axs = plt.subplots(1, num_stages, figsize=(5 * num_stages, 5))
    for i, (title, image) in enumerate(stages):
        if len(stages) == 1:
            axs.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
            axs.set_title(title)
            axs.axis('off')
        else:
            axs[i].imshow(image, cmap="gray" if len(image.shape) == 2 else None)
            axs[i].set_title(title)
            axs[i].axis('off')
    plt.show()

# Procesar todas las imágenes en las carpetas
for folder in ["train", "test", "val"]:
    input_folder = os.path.join(input_dir, folder)
    output_folder = os.path.join(output_dir, folder)
    
    # Crear subcarpeta en salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterar sobre las subcarpetas de cada clase
    for class_name in os.listdir(input_folder):
        class_input_folder = os.path.join(input_folder, class_name)
        class_output_folder = os.path.join(output_folder, class_name)
        
        # Crear subcarpeta para la clase en salida si no existe
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)
        
        # Procesar imágenes dentro de la subcarpeta
        for file_name in os.listdir(class_input_folder):
            if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                input_path = os.path.join(class_input_folder, file_name)
                output_path = os.path.join(class_output_folder, file_name)
                
                # Procesar la imagen
                processed_image, stages = process_image(input_path)
                
                # Guardar la imagen procesada en color
                if processed_image is not None:
                    cv2.imwrite(output_path, processed_image)
                
                # Mostrar todas las etapas de procesamiento al final
                if stages is not None:
                    show_all_stages(stages)

print("Procesamiento completado. Las imágenes procesadas en color se encuentran en:", output_dir)
