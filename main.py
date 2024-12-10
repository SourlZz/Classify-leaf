import cv2
import os
import numpy as np

# Define las rutas de entrada y salida
input_dir = "./archive/split"  # Carpeta raíz con subcarpetas 'train', 'test', y 'val'
output_dir = "./split_processed"  # Carpeta donde se guardarán las imágenes procesadas

# Crear directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Función para procesar imágenes
def process_image(image_path):
    # Leer la imagen
    image = cv2.imread(image_path)
    
    # Verificar si la imagen se ha cargado correctamente
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None
    
    # Convertir a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Aplicar un suavizado más ligero
    blurred_image = cv2.GaussianBlur(hsv_image, (3, 3), 0)  # Kernel ajustado a (3, 3)
    
    # Separar canales HSV
    h, s, v = cv2.split(blurred_image)
    
    # Ecualizar el histograma del canal de brillo (V)
    v_equalized = cv2.equalizeHist(v)
    
    # Aplicar contraste moderado al canal V
    v_contrast = cv2.normalize(v_equalized, None, alpha=50, beta=200, norm_type=cv2.NORM_MINMAX)
    
    # Detección de bordes menos agresiva
    edges = cv2.Canny(v_contrast, 20, 60)
    
    # Combinar bordes con el canal V
    v_combined = cv2.addWeighted(v_contrast, 0.9, edges, 0.1, 0)
    
    # Recomponer la imagen en el espacio HSV con el canal V procesado
    processed_hsv = cv2.merge([h, s, v_combined])
    
    # Convertir de HSV a BGR para guardar la imagen procesada en color
    processed_bgr = cv2.cvtColor(processed_hsv, cv2.COLOR_HSV2BGR)
    
    return processed_bgr

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
                processed_image = process_image(input_path)
                
                # Guardar la imagen procesada en color
                if processed_image is not None:
                    cv2.imwrite(output_path, processed_image)

print("Procesamiento completado. Las imágenes procesadas en color se encuentran en:", output_dir)
