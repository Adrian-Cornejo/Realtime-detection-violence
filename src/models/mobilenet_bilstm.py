import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# Configuración
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
SEQUENCE_LENGTH = 20

def create_mobilenet_base():
    """
    Crea el modelo base MobileNetV2 con fine-tuning.
    
    Returns:
        Modelo base MobileNetV2 configurado
    """
    # Cargar MobileNetV2 con la forma de entrada correcta
    mobilenet = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    )
    
    # Fine-Tuning con más capas descongeladas (50 en lugar de 40)
    mobilenet.trainable = True
    for layer in mobilenet.layers[:-50]:  # Congelamos primeras capas
        layer.trainable = False
        
    return mobilenet

def create_model(num_classes=2):
    """
    Crea el modelo completo MobileNetV2 + BiLSTM.
    
    Args:
        num_classes: Número de clases para la capa de salida
        
    Returns:
        Modelo secuencial configurado
    """
    # Cargar modelo base
    mobilenet = create_mobilenet_base()
    
    # Construir el modelo
    model = Sequential(name="Violence_Detection_MoBiLSTM")
    
    # Entrada
    model.add(Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name="input_layer"))
    
    # Aplicar el modelo base a cada frame
    model.add(TimeDistributed(mobilenet, name="mobilenetv2_feature_extractor"))
    
    # Global Average Pooling para reducir parámetros
    model.add(TimeDistributed(GlobalAveragePooling2D(), name="global_avg_pooling"))
    model.add(Dropout(0.3, name="dropout_1"))
    
    # BiLSTM mejorado
    model.add(Bidirectional(LSTM(units=128, return_sequences=True), name="bidirectional_lstm"))
    model.add(Dropout(0.3, name="dropout_2"))
    
    # Segundo LSTM para capturar dependencias temporales más complejas
    model.add(LSTM(units=128, name="lstm_layer"))
    model.add(Dropout(0.3, name="dropout_3"))
    
    # Capas densas con normalización
    model.add(Dense(256, activation='relu', name="dense_1"))
    model.add(BatchNormalization(name="batch_norm_1"))
    model.add(Dropout(0.4, name="dropout_4"))
    
    model.add(Dense(128, activation='relu', name="dense_2"))
    model.add(BatchNormalization(name="batch_norm_2"))
    model.add(Dropout(0.3, name="dropout_5"))
    
    # Capa de salida
    model.add(Dense(num_classes, activation='softmax', name="output_layer"))
    
    return model

def compile_model(model, learning_rate=0.0001):
    """
    Compila el modelo con configuración óptima.
    
    Args:
        model: Modelo a compilar
        learning_rate: Tasa de aprendizaje para el optimizador
        
    Returns:
        Modelo compilado
    """
    # Compilar modelo con Adam y métricas completas
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    return model

def get_callbacks(patience=15, min_lr=0.00001, model_path="models/saved_models"):
    """
    Crea callbacks para el entrenamiento.
    
    Args:
        patience: Paciencia para EarlyStopping
        min_lr: Learning rate mínimo para ReduceLROnPlateau
        model_path: Ruta para guardar el modelo
        
    Returns:
        Lista de callbacks
    """
    os.makedirs(model_path, exist_ok=True)
    
    callbacks = [
        # Detención temprana
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Reducción de learning rate
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=1
        ),
        # Guardar mejor modelo
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_path, 'violence_detection_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

def load_model(model_path="models/saved_models/violence_detection_model.h5"):
    """
    Carga un modelo previamente entrenado.
    
    Args:
        model_path: Ruta al archivo del modelo
        
    Returns:
        Modelo cargado
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Modelo cargado desde: {model_path}")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None