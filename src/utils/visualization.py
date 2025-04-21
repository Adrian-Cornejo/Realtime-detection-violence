import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score

def plot_training_curves(history):
    """
    Visualiza las curvas de entrenamiento del modelo.
    
    Args:
        history: Objeto history devuelto por model.fit()
    """
    # Crear figura con subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    
    # Precisión
    axs[0, 0].plot(history.history['accuracy'], label='Train')
    axs[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axs[0, 0].set_title('Precisión')
    axs[0, 0].set_ylabel('Precisión')
    axs[0, 0].set_xlabel('Época')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Pérdida
    axs[0, 1].plot(history.history['loss'], label='Train')
    axs[0, 1].plot(history.history['val_loss'], label='Validation')
    axs[0, 1].set_title('Pérdida')
    axs[0, 1].set_ylabel('Pérdida')
    axs[0, 1].set_xlabel('Época')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Precision (si está disponible)
    if 'precision' in history.history:
        axs[1, 0].plot(history.history['precision'], label='Train')
        axs[1, 0].plot(history.history['val_precision'], label='Validation')
        axs[1, 0].set_title('Precision')
        axs[1, 0].set_ylabel('Precision')
        axs[1, 0].set_xlabel('Época')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
    
    # Recall (si está disponible)
    if 'recall' in history.history:
        axs[1, 1].plot(history.history['recall'], label='Train')
        axs[1, 1].plot(history.history['val_recall'], label='Validation')
        axs[1, 1].set_title('Recall')
        axs[1, 1].set_ylabel('Recall')
        axs[1, 1].set_xlabel('Época')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Visualiza la matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        class_names: Nombres de las clases
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Verdad')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()
    
    # Calcular métricas por clase
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"Verdaderos Negativos (No Violencia correctamente identificada): {tn}")
    print(f"Falsos Positivos (No Violencia clasificada incorrectamente como Violencia): {fp}")
    print(f"Falsos Negativos (Violencia clasificada incorrectamente como No Violencia): {fn}")
    print(f"Verdaderos Positivos (Violencia correctamente identificada): {tp}")
    print(f"\nSensibilidad (Recall): {sensitivity:.4f}")
    print(f"Especificidad: {specificity:.4f}")
    print(f"Precisión: {precision:.4f}")

def plot_roc_curve(y_true, y_pred_prob):
    """
    Visualiza la curva ROC.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred_prob: Probabilidades predichas para la clase positiva
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_prob):
    """
    Visualiza la curva precision-recall.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred_prob: Probabilidades predichas para la clase positiva
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

def display_sample_frames(video_path, frames_to_show=4):
    """
    Muestra frames de muestra de un video.
    
    Args:
        video_path: Ruta al video
        frames_to_show: Número de frames a mostrar
    """
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    
    # Verificar que se haya abierto correctamente
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return
    
    # Obtener información del video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    print(f"Video: {video_path}")
    print(f"Frames totales: {frame_count}")
    print(f"FPS: {fps}")
    print(f"Duración: {duration:.2f} segundos")
    
    # Seleccionar frames para mostrar
    frame_indices = [int(i * frame_count / (frames_to_show + 1)) for i in range(1, frames_to_show + 1)]
    
    plt.figure(figsize=(15, 5))
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error al leer el frame {frame_index}")
            continue
        
        # Convertir de BGR a RGB para matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, frames_to_show, i + 1)
        plt.imshow(frame_rgb)
        plt.title(f"Frame {frame_index}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Liberar recursos
    cap.release()