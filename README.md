# Procesamiento de Imágenes con OpenCV

Este proyecto procesa y mejora imágenes almacenadas en diferentes carpetas utilizando OpenCV en Python.

## Requisitos

Antes de ejecutar el script, asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install opencv-python numpy
```

## Estructura de Carpetas

El script espera una estructura de carpetas de entrada con imágenes organizadas en subdirectorios de entrenamiento, prueba y validación:

```
./archive/split/
    ├── train/
    │   ├── clase1/
    │   │   ├── imagen1.jpg
    │   │   ├── imagen2.jpg
    │   └── ...
    ├── test/
    │   ├── clase1/
    │   ├── clase2/
    │   └── ...
    ├── val/
    │   ├── clase1/
    │   ├── clase2/
    │   └── ...
```

Las imágenes procesadas se guardarán en una estructura de salida similar dentro de la carpeta `./split_processed/`.

## Funcionamiento del Script

1. **Carga la imagen** utilizando OpenCV.
2. **Convierte la imagen a HSV** para mejorar el procesamiento de brillo y contraste.
3. **Aplica un suavizado** con un filtro Gaussiano para reducir el ruido.
4. **Ecualiza el histograma del canal de brillo (V)** para mejorar el contraste.
5. **Aplica detección de bordes con Canny** con valores moderados.
6. **Combina los bordes con el canal V** para resaltar detalles importantes.
7. **Reconstruye la imagen en BGR** y la guarda en la carpeta de salida.

## Ejecución

Para ejecutar el script, simplemente corre:

```bash
python script.py
```

Al finalizar, las imágenes procesadas estarán disponibles en `./split_processed/` con la misma organización de carpetas que la entrada.

## Notas

- Si alguna imagen no se puede cargar correctamente, se imprimirá un mensaje de error sin detener la ejecución del script.
- Se recomienda ajustar los valores de detección de bordes y normalización de contraste si se necesita una configuración diferente.


