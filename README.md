# Sistema de Detección de Violencia en Tiempo Real

Un sistema de detección de violencia basado en aprendizaje profundo que utiliza una arquitectura MobileNetV2 + BiLSTM para identificar situaciones de violencia en secuencias de video.

## Descripción

Este proyecto implementa un modelo de clasificación de secuencias de vídeo en tiempo real que puede distinguir entre situaciones violentas y no violentas. Utiliza técnicas de deep learning aplicando transfer learning con MobileNetV2 como extractor de características y una arquitectura BiLSTM para procesar la información temporal en las secuencias de fotogramas.

## Características

- **Alta precisión**: ~94% de precisión en la detección de situaciones de violencia
- **Procesamiento en tiempo real**: Optimizado para funcionar con secuencias de video en vivo
- **Arquitectura eficiente**: Basada en MobileNetV2 preentrenado con ImageNet para extracción eficiente de características
- **Capacidades temporales**: Uso de BiLSTM para capturar patrones temporales en las secuencias de video
- **Interfaz visual**: Incluye marcadores visuales para identificar la violencia detectada

## Arquitectura del Modelo

La arquitectura combina:

1. **MobileNetV2**: Red convolucional preentrenada que funciona como extractor de características espaciales
2. **BiLSTM**: Capa LSTM bidireccional que captura patrones temporales en las secuencias
3. **Capas densas**: Con normalización por lotes para la clasificación final

## Conjunto de datos

El modelo fue entrenado usando el "Real Life Violence Situations Dataset" que contiene clips de video de situaciones violentas y no violentas de la vida real.

- **Clases**: Violencia y No-Violencia
- **Resolución de entrada**: 96x96 píxeles
- **Longitud de secuencia**: 20 frames

## Resultados

El modelo logra:

- **Precisión en entrenamiento**: ~94%
- **Precisión en validación**: ~93.8%
- **Pérdida final**: ~0.15

## Requisitos

```
tensorflow>=2.4.0
opencv-python>=4.5.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/deteccion-violencia.git
cd deteccion-violencia

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Entrenamiento del modelo

```python
# Definir hiperparámetros
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
SEQUENCE_LENGTH = 20
BATCH_SIZE = 8

# Preprocesar datos
features, labels = create_dataset(augment=True)

# Crear y entrenar modelo
model = create_model()
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

model.fit(x=features_train, y=labels_train,
          epochs=50, batch_size=BATCH_SIZE,
          validation_split=0.2)
```

### Detección en tiempo real

```python
# Cargar modelo entrenado
model = tf.keras.models.load_model('violence_detection_model.h5')

# Detectar en video
detect_on_video(model, "input_video.mp4", "output_video.mp4")

# Detectar en webcam
detect_on_webcam(model)
```

## Extensiones futuras

- **Implementación en ESP32**: Captura de imágenes y transmisión a un servidor para procesamiento
- **Sistema de alertas**: Notificaciones por email, SMS o aplicación móvil ante detección de violencia
- **Ampliación de clases**: Reconocimiento de tipos específicos de violencia
- **Mejora de rendimiento**: Optimización para dispositivos con recursos limitados

## Licencia

Este proyecto está licenciado bajo la licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Agradecimientos

- Dataset proporcionado por Mohamed Mustafa en Kaggle
- Base del modelo MobileNetV2 desarrollada por Google
