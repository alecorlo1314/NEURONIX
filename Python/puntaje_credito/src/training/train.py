"""
Módulo de Entrenamiento para MLP de Puntaje de Crédito
Este módulo contiene la clase principal para entrenar modelos de redes neuronales
para la predicción de puntaje de crédito.
"""
# Importaciones del sistema operativo y utilidades
import os  # Para operaciones del sistema operativo (rutas, variables de entorno)
import sys  # Para acceso a funciones y variables del sistema Python
import yaml  # Para leer archivos de configuración en formato YAML
import math  # Para operaciones matemáticas (verificar NaN, etc.)
import argparse  # Para parsear argumentos de línea de comandos
import warnings  # Para suprimir advertencias específicas durante la ejecución

# Importaciones de PyTorch para deep learning
import torch  # Framework principal de deep learning
import torch.nn as nn  # Módulos de redes neuronales (capas, funciones de activación)
import torch.optim as optim  # Optimizadores (Adam, SGD, etc.)
import logging as log  # Sistema de logging para registrar eventos y errores

# Importaciones de MLflow para experimentación y tracking
import mlflow  # Framework para tracking de experimentos de ML
import mlflow.pytorch  # Integración específica de MLflow con PyTorch

# Importaciones de procesamiento de datos
import numpy as np  # Operaciones numéricas y arrays multidimensionales
import pandas as pd  # Manipulación y análisis de datos estructurados (DataFrames)
import joblib  # Para guardar y cargar objetos de Python (preprocesadores, modelos)

# Importaciones de visualización
import matplotlib.pyplot as plt  # Para crear gráficos y visualizaciones

# Importaciones de utilidades
from pathlib import Path  # Para manejo de rutas de archivos de forma multiplataforma
from datetime import datetime, timezone  # Para timestamps y fechas
from typing import Tuple, Dict, Any, List  # Para anotaciones de tipos en Python

# Importaciones de scikit-learn para evaluación y división de datos
from sklearn.model_selection import train_test_split  # Para dividir datos en entrenamiento y validación
from sklearn.metrics import (
    accuracy_score,  # Métrica de precisión (exactitud)
    roc_auc_score,  # Área bajo la curva ROC
    precision_recall_fscore_support,  # Precisión, recall, F1-score y soporte
    roc_curve,  # Para generar curva ROC
    precision_recall_curve,  # Para generar curva Precision-Recall
    confusion_matrix,  # Matriz de confusión
    classification_report  # Reporte de clasificación detallado
)

# Importación de MLflow para inferir la firma del modelo
from mlflow.models.signature import infer_signature  # Para inferir la firma de entrada/salida del modelo

# Agregar el directorio raíz del proyecto al path de Python para importar módulos locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Importar el preprocesador de datos personalizado
from src.processing.main import CreditDataPreprocessor
# Importar el modelo de red neuronal personalizado
from src.training.model import CreditScoringModel


def setup_logging(level=log.INFO, log_file: str | None = None):
    """
    Configura el sistema de logging para el entrenamiento.
    
    Args:
        level: Nivel de logging (INFO, DEBUG, WARNING, ERROR)
        log_file: Ruta opcional al archivo donde guardar los logs
    """
    # Crear lista de handlers (destinos de los mensajes de log)
    handlers = [log.StreamHandler(sys.stdout)]  # Handler para imprimir en consola
    
    # Si se especifica un archivo, agregar un handler rotativo para archivos
    if log_file:
        from logging.handlers import RotatingFileHandler
        # RotatingFileHandler rota el archivo cuando alcanza 5MB, mantiene 3 backups
        handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))

    # Configurar el sistema de logging con formato personalizado
    log.basicConfig(
        level=level,  # Nivel mínimo de logging
        format="%(asctime)s - %(levelname)s - %(message)s",  # Formato: fecha-hora - nivel - mensaje
        handlers=handlers,  # Usar los handlers definidos
        force=True,  # Forzar reconfiguración incluso si ya estaba configurado
    )
    # Reducir el nivel de logging de librerías externas para evitar ruido innecesario
    for noisy in ("mlflow", "urllib3", "matplotlib"):
        log.getLogger(noisy).setLevel(log.WARNING)


class CreditScoringModelTraining:
    """
    Clase principal para entrenar modelos de puntaje de crédito.
    Maneja todo el pipeline de entrenamiento: carga de datos, preprocesamiento,
    entrenamiento del modelo, evaluación y logging en MLflow.
    """
    def __init__(self, config_path: Path) -> None:
        """
        Inicializa la clase con la configuración desde un archivo YAML.
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        # Abrir y cargar el archivo de configuración YAML
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)  # Cargar parámetros desde YAML de forma segura
        
        log.info(f"--- Configuracion del Entrenamiento ---")
        
        # ========== CONFIGURACIÓN DE RUTAS ==========
        # Ruta al archivo CSV con los datos de entrenamiento
        self.dataset_path = Path(self.params['data_source']['data_path']['dataset_path'])
        # Nombre del artefacto del modelo en MLflow (ej: "model")
        self.artifact_name_or_path = self.params['data_source']['data_path']['artifact_path']
        # Nombre del archivo del preprocesador guardado (ej: "preprocessor.joblib")
        self.preprocessor_filename = self.params['data_source']['data_path']['preprocessor_filename']
        
        # ========== CONFIGURACIÓN DE ARQUITECTURA DEL MODELO ==========
        model_cfg = self.params['model_config']['architecture']
        self.hidden_layers = model_cfg['hidden_layers']  # Lista con número de neuronas por capa oculta
        self.use_batch_norm = model_cfg['use_batch_norm']  # Si usar normalización por lotes
        self.activation_fn = model_cfg['activation_fn']  # Función de activación (ReLU, Tanh, etc.)
        self.dropout_rate = model_cfg['dropout_rate']  # Tasa de dropout para regularización
        
        # ========== CONFIGURACIÓN DE ENTRENAMIENTO ==========
        train_cfg = self.params['training_params']
        self.optimizer_name = train_cfg['optimizer']['name']  # Nombre del optimizador (Adam, SGD, etc.)
        self.learning_rate = train_cfg['optimizer']['learning_rate']  # Tasa de aprendizaje
        self.weight_decay = train_cfg['optimizer'].get('weight_decay', 0.0)  # Decaimiento de pesos (L2 regularization), default 0
        self.use_pos_weight = train_cfg['loss_function']['use_pos_weight']  # Si usar peso para clase positiva (balanceo)
        self.scheduler_patience = train_cfg['scheduler']['patience']  # Paciencia del scheduler (épocas sin mejora)
        self.scheduler_factor = train_cfg['scheduler']['factor']  # Factor de reducción del learning rate
        self.epochs = train_cfg['epochs']  # Número máximo de épocas de entrenamiento
        self.batch_size = train_cfg['batch_size']  # Tamaño del lote (batch size)
        
        # ========== CONFIGURACIÓN DE DATOS ==========
        self.test_size = train_cfg['test_size']  # Proporción de datos para validación (ej: 0.2 = 20%)
        self.random_state = train_cfg['random_state']  # Semilla aleatoria para reproducibilidad
        
        # ========== CONFIGURACIÓN DE EARLY STOPPING ==========
        self.early_stopping_patience = train_cfg['early_stopping']['patience']  # Épocas sin mejora antes de detener
        self.early_stopping_delta = train_cfg['early_stopping']['delta']  # Mejora mínima requerida para considerar progreso
        
        # ========== CONFIGURACIÓN DE NOMBRES ==========
        self.model_name = self.params['model_config']['model_name']  # Nombre del archivo del modelo guardado
        self.mlflow_project_name = self.params['mlflow_config']['mlflow_project_name']  # Nombre del proyecto en MLflow
        
        # ========== REPRODUCIBILIDAD ==========
        # Establecer semillas aleatorias para garantizar resultados reproducibles
        np.random.seed(self.random_state)  # Semilla para NumPy
        torch.manual_seed(self.random_state)  # Semilla para PyTorch
        
        # ========== CONFIGURACIÓN DE HARDWARE ==========
        # Detectar y usar GPU si está disponible, sino usar CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ========== INSTANCIAS DE COMPONENTES ==========
        # Crear instancia del preprocesador de datos
        self.data_preprocessor = CreditDataPreprocessor()
        
        # ========== HISTORIAL DE ENTRENAMIENTO ==========
        # Diccionario para almacenar métricas durante el entrenamiento
        self.history: Dict[str, List[float]] = {
            "train_loss": [],  # Pérdida en entrenamiento por época
            "val_loss": [],    # Pérdida en validación por época
            "train_acc": [],   # Precisión en entrenamiento por época
            "val_acc": [],     # Precisión en validación por época
            "train_auc": [],   # AUC-ROC en entrenamiento por época
            "val_auc": []      # AUC-ROC en validación por época
        }
        
        # ========== CARPETA DE ARTEFACTOS ==========
        # Crear directorio para guardar reportes y gráficos localmente
        self.local_artifacts_dir = Path("reports")
        self.local_artifacts_dir.mkdir(parents=True, exist_ok=True)  # Crear si no existe
        
    def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga los datos desde el archivo CSV y los divide en conjuntos de entrenamiento y validación.
        
        Returns:
            Tupla con (DataFrame de entrenamiento, DataFrame de validación)
        """
        log.info(f"--- Cargar datos ---")
        log.info(f"✔ Cargando datos desde {self.dataset_path}")
        
        # Intentar cargar el archivo CSV
        try:
            df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            log.error(f"Archivo no funciona en: {self.dataset_path}")
            raise  # Re-lanzar la excepción para detener la ejecución
        
        # Eliminar columna 'Unnamed: 0' si existe (suele aparecer al guardar CSV con índice)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            log.info("✔ Columna 'Unnamed: 0' eliminada del DataFrame.")
        
        # Dividir los datos en entrenamiento y validación
        log.info("✔ Separando datos en entrenamiento y validacion")
        df_train, df_val = train_test_split(
            df,  # DataFrame completo
            test_size=self.test_size,  # Proporción para validación (ej: 0.2 = 20%)
            random_state=self.random_state,  # Semilla para reproducibilidad
            stratify=df[self.data_preprocessor.target_feature]  # Mantener proporción de clases en ambos conjuntos
        )
        return df_train, df_val
    
    def _preprocess_data(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Tuple[torch.Tensor, ...]:
        """
        Ajusta el preprocesador con los datos de entrenamiento y transforma ambos conjuntos.
        Convierte los datos a tensores de PyTorch y guarda el preprocesador ajustado.
        
        Args:
            df_train: DataFrame con datos de entrenamiento
            df_val: DataFrame con datos de validación
            
        Returns:
            Tupla con (X_train, y_train, X_val, y_val) como tensores de PyTorch
        """
        log.info("--- Preprocesamiento de Datos ---")
        
        # Ajustar el preprocesador solo con datos de entrenamiento (fit)
        # Esto aprende las transformaciones necesarias (escalado, encoding, etc.)
        preprocessor = self.data_preprocessor.fit_preprocessor(df_train)
        
        # Transformar ambos conjuntos usando el preprocesador ajustado
        x_train_processed, y_train = self.data_preprocessor.process_data(df_train, preprocessor)
        x_val_processed, y_val = self.data_preprocessor.process_data(df_val, preprocessor)
        
        # Convertir arrays de NumPy a tensores de PyTorch
        # X_train: características de entrenamiento como tensor de float32
        x_train_tensor = torch.tensor(x_train_processed, dtype=torch.float32)
        # y_train: etiquetas de entrenamiento como tensor de float32, reshape a columna
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        # X_val: características de validación como tensor de float32
        x_val_tensor = torch.tensor(x_val_processed, dtype=torch.float32)
        # y_val: etiquetas de validación como tensor de float32, reshape a columna
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        
        # Guardar el preprocesador ajustado para uso futuro en inferencia
        path_preprocessor = f"models/{self.preprocessor_filename}"
        joblib.dump(preprocessor, path_preprocessor)  # Serializar y guardar el preprocesador
        log.info(f"✔ preprocesamiento guardado en {path_preprocessor}")
        
        return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor
    
    # ========== MÉTRICAS ==========
    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calcula métricas de evaluación del modelo: accuracy, precision, recall, F1 y ROC-AUC.
        
        Args:
            y_true: Array con las etiquetas verdaderas (N,)
            y_prob: Array con las probabilidades predichas de clase positiva (N,)
            threshold: Umbral para convertir probabilidades a predicciones binarias (default: 0.5)
            
        Returns:
            Diccionario con las métricas calculadas
        """
        # Convertir probabilidades a predicciones binarias usando el umbral
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calcular accuracy (precisión/exactitud): proporción de predicciones correctas
        acc = accuracy_score(y_true, y_pred)
        
        # Calcular precision, recall, F1-score y soporte para clasificación binaria
        # average='binary' indica que es un problema de clasificación binaria
        # zero_division=0 maneja casos donde no hay predicciones positivas
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        # Calcular AUC-ROC (área bajo la curva ROC)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Si hay error (ej: solo una clase presente), retornar NaN
            auc = float('nan')
        
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
    
    def _evaluate_split(self, model, x: torch.Tensor, y: torch.Tensor, criterion) -> Dict[str, float]:
        """
        Evalúa un conjunto completo de datos (entrenamiento o validación) y calcula pérdida y métricas.
        
        Args:
            model: Modelo de PyTorch a evaluar
            x: Tensor con las características de entrada
            y: Tensor con las etiquetas verdaderas
            criterion: Función de pérdida (loss function)
            
        Returns:
            Diccionario con pérdida y todas las métricas calculadas
        """
        # Poner el modelo en modo evaluación (desactiva dropout, batch norm en modo eval, etc.)
        model.eval()
        
        # Desactivar cálculo de gradientes para evaluación (ahorra memoria y velocidad)
        with torch.no_grad():
            # Obtener predicciones del modelo (logits, antes de sigmoid)
            logits = model(x)
            
            # Calcular la pérdida
            loss = criterion(logits, y).item()  # .item() convierte tensor escalar a Python float
            
            # Convertir logits a probabilidades usando sigmoid y mover a CPU como numpy array
            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            
            # Convertir etiquetas a numpy array
            y_true = y.detach().cpu().numpy().reshape(-1)
            
            # Calcular métricas usando el método estático
            m = self._compute_metrics(y_true, prob, threshold=0.5)
            
            # Agregar la pérdida al diccionario de métricas
            m["loss"] = loss
        
        return m
    
    # ========== VISUALIZACIONES ==========
    def _plot_and_save(self, xs: List[int], ys1: List[float], ys2: List[float], title: str, ylabel: str, filename: str):
        """
        Crea un gráfico de líneas comparando entrenamiento vs validación y lo guarda.
        
        Args:
            xs: Lista de épocas (eje X)
            ys1: Valores de entrenamiento (eje Y)
            ys2: Valores de validación (eje Y)
            title: Título del gráfico
            ylabel: Etiqueta del eje Y
            filename: Nombre del archivo donde guardar el gráfico
            
        Returns:
            Ruta al archivo guardado
        """
        # Crear una nueva figura
        plt.figure()
        
        # Graficar línea de entrenamiento
        plt.plot(xs, ys1, label="train")
        # Graficar línea de validación
        plt.plot(xs, ys2, label="val")
        
        # Configurar etiquetas de ejes
        plt.xlabel("Epoch")  # Eje X: épocas
        plt.ylabel(ylabel)   # Eje Y: métrica específica (Loss, Accuracy, etc.)
        plt.title(title)     # Título del gráfico
        
        # Mostrar leyenda
        plt.legend()
        
        # Construir ruta completa del archivo de salida
        out = self.local_artifacts_dir / filename
        
        # Guardar figura con ajuste automático de bordes
        plt.savefig(out, bbox_inches="tight")
        
        # Cerrar la figura para liberar memoria
        plt.close()
        
        return out
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, filename: str):
        """
        Crea y guarda una visualización de la matriz de confusión.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Etiquetas predichas
            filename: Nombre del archivo donde guardar el gráfico
            
        Returns:
            Ruta al archivo guardado
        """
        # Calcular matriz de confusión con etiquetas [0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        
        # Crear nueva figura
        plt.figure()
        
        # Mostrar matriz como imagen con interpolación
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix (val)")
        
        # Agregar barra de color para escala
        plt.colorbar()
        
        # Configurar marcas de los ejes
        tick_marks = [0,1]
        plt.xticks(tick_marks, ['bad(0)','good(1)'])  # Etiquetas del eje X
        plt.yticks(tick_marks, ['bad(0)','good(1)'])  # Etiquetas del eje Y
        
        # Agregar valores numéricos dentro de cada celda de la matriz
        # Calcular umbral para determinar color del texto (blanco o negro)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        
        # Iterar sobre cada celda de la matriz
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Agregar texto con el valor de la celda
                plt.text(j, i, format(cm[i, j], 'd'),  # 'd' = formato decimal
                        ha="center", va="center",  # Centrar texto
                        color="white" if cm[i, j] > thresh else "black")  # Color según umbral
        
        # Etiquetas de ejes
        plt.ylabel('Clase Verdadera')
        plt.xlabel('Clase Predicha')
        
        # Guardar figura
        out = self.local_artifacts_dir / filename
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        
        return out
    
    def _plot_roc_pr(self, y_true: np.ndarray, y_prob: np.ndarray, roc_file: str, pr_file: str):
        """
        Crea y guarda gráficos de curva ROC y Precision-Recall.
        
        Args:
            y_true: Etiquetas verdaderas
            y_prob: Probabilidades predichas de clase positiva
            roc_file: Nombre del archivo para la curva ROC
            pr_file: Nombre del archivo para la curva Precision-Recall
            
        Returns:
            Tupla con rutas a los archivos guardados (pueden ser None si hay error)
        """
        # ========== CURVA ROC ==========
        try:
            # Calcular puntos de la curva ROC (False Positive Rate, True Positive Rate)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            
            # Crear nueva figura
            plt.figure()
            
            # Graficar curva ROC
            plt.plot(fpr, tpr)
            
            # Graficar línea diagonal (clasificador aleatorio de referencia)
            plt.plot([0,1],[0,1], linestyle='--')
            
            # Configurar etiquetas y título
            plt.xlabel("Tasa de falsos positivos")
            plt.ylabel("Tasa de verdaderos positivos")
            plt.title("Curva ROC (val)")
            
            # Guardar figura
            roc_path = self.local_artifacts_dir / roc_file
            plt.savefig(roc_path, bbox_inches="tight")
            plt.close()
        except ValueError:
            # Si hay error (ej: solo una clase presente), no guardar gráfico
            roc_path = None

        # ========== CURVA PRECISION-RECALL ==========
        try:
            # Calcular puntos de la curva Precision-Recall
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            
            # Crear nueva figura
            plt.figure()
            
            # Graficar curva Precision-Recall
            plt.plot(rec, prec)
            
            # Configurar etiquetas y título
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve (val)")
            
            # Guardar figura
            pr_path = self.local_artifacts_dir / pr_file
            plt.savefig(pr_path, bbox_inches="tight")
            plt.close()
        except ValueError:
            # Si hay error (ej: solo una clase presente), no guardar gráfico
            pr_path = None

        return roc_path, pr_path
    
    def _run_training_loop(self, model, criterion, optimizer, scheduler, x_train, y_train, x_val, y_val):
        """
        Ejecuta el ciclo principal de entrenamiento y validación con early stopping.
        
        Args:
            model: Modelo de PyTorch a entrenar
            criterion: Función de pérdida
            optimizer: Optimizador (Adam, SGD, etc.)
            scheduler: Scheduler para ajustar learning rate
            x_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            x_val: Características de validación
            y_val: Etiquetas de validación
            
        Returns:
            Número de épocas ejecutadas
        """
        # Inicializar variables para early stopping
        best_val_loss = float('inf')  # Mejor pérdida de validación encontrada
        patience_counter = 0  # Contador de épocas sin mejora
        epochs_run = 0  # Número de épocas ejecutadas
        
        log.info("--- Iniciando el ciclo de entrenamiento ---")
        
        # Ciclo principal de entrenamiento por épocas
        for epoch in range(self.epochs):
            # Poner modelo en modo entrenamiento (activa dropout, batch norm en modo train)
            model.train()
            epoch_loss = 0  # Acumulador de pérdida para esta época
            epochs_run = epoch + 1  # Actualizar contador de épocas
            
            # ========== ENTRENAMIENTO POR MINI-LOTES ==========
            # Procesar datos en lotes (batches) para eficiencia de memoria
            for i in range(0, len(x_train), self.batch_size):
                # Extraer un lote de datos
                x_batch = x_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]
                
                # Forward pass: obtener predicciones del modelo
                outputs = model(x_batch)
                
                # Calcular pérdida comparando predicciones con etiquetas verdaderas
                loss = criterion(outputs, y_batch)
                
                # Backward pass: calcular gradientes
                optimizer.zero_grad()  # Limpiar gradientes acumulados de iteraciones anteriores
                loss.backward()  # Calcular gradientes mediante backpropagation
                
                # Recortar gradientes para evitar explosión de gradientes (gradient clipping)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Actualizar pesos del modelo usando los gradientes calculados
                optimizer.step()
                
                # Acumular pérdida del lote
                epoch_loss += loss.item()
                
            # ========== EVALUACIÓN ==========
            # Evaluar modelo en conjunto de entrenamiento completo
            train_metrics = self._evaluate_split(model, x_train, y_train, criterion)
            # Evaluar modelo en conjunto de validación completo
            val_metrics = self._evaluate_split(model, x_val, y_val, criterion)
            
            # Ajustar learning rate si hay scheduler configurado
            if scheduler is not None:
                scheduler.step(val_metrics["loss"])  # Reduce LR si no hay mejora
                
            # ========== GUARDAR HISTORIAL ==========
            # Almacenar métricas en el historial para visualización posterior
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["train_auc"].append(train_metrics["roc_auc"])
            self.history["val_auc"].append(val_metrics["roc_auc"])
            
            # ========== LOGGING A CONSOLA ==========
            # Obtener learning rate actual
            current_lr = optimizer.param_groups[0]["lr"]
            # Imprimir métricas de la época actual
            log.info(
                f"✔ Epoca [{epoch}/{self.epochs}] "
                f"✔ PerdidaEntrenamiento: {train_metrics['loss']:.4f} | ValLoss: {val_metrics['loss']:.4f} | "
                f"✔ TrainAcc: {train_metrics['accuracy']:.4f} | ValAcc: {val_metrics['accuracy']:.4f} | "
                f"✔ TrainAUC: {train_metrics['roc_auc']:.4f} | ValAUC: {val_metrics['roc_auc']:.4f} | LR: {current_lr:.6f}"
            )
            
            # ========== LOGGING A MLFLOW ==========
            # Registrar todas las métricas en MLflow para tracking de experimentos
            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "val_precision": val_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "val_recall": val_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "train_roc_auc": train_metrics["roc_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "lr": current_lr  # Learning rate actual
            }, step=epoch)  # step indica la época para el tracking temporal

            # ========== EARLY STOPPING ==========
            # Verificar si hay mejora significativa en la pérdida de validación
            if val_metrics["loss"] < best_val_loss - self.early_stopping_delta:
                # Hay mejora: actualizar mejor pérdida y resetear contador
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                # Guardar el mejor modelo hasta ahora
                path_model = f"models/{self.model_name}"
                torch.save(model.state_dict(), path_model)  # Guardar solo los pesos del modelo
            else:
                # No hay mejora: incrementar contador de paciencia
                patience_counter += 1
                # Si se alcanza la paciencia máxima, detener entrenamiento
                if patience_counter >= self.early_stopping_patience:
                    log.info(f"✘ Early stopping activado en época {epoch}.")
                    break
                
        log.info("--- Entrenamiento terminado ---")
        return epochs_run
        
    # ========== MLFLOW ==========
    def _generate_and_log_performance_report(
        self,
        model: CreditScoringModel,
        final_metrics: Dict[str, float],
        num_features: int,
        epochs_run: int,
        run_name: str
    ):
        """
        Genera un reporte de rendimiento en formato YAML, lo guarda localmente y lo registra en MLflow.
        
        Args:
            model: Modelo entrenado
            final_metrics: Métricas finales de validación
            num_features: Número de características de entrada
            epochs_run: Número de épocas ejecutadas
            run_name: Nombre del run de MLflow
        """
        log.info("--- Generando informe de rendimiento ---")
        
        # Obtener información detallada del modelo
        model_info = model.get_model_info()
        
        # Construir diccionario con toda la información del reporte
        report_data = {
            "benchmark_id": self.params.get("project_info", {}).get("benchmark_id", "N/A"),  # ID del benchmark
            "timestamp": datetime.now(timezone.utc).isoformat(),  # Timestamp en UTC
            "model_architecture": {
                "model_type": model_info["model_type"],  # Tipo de modelo
                "input_features": num_features,  # Número de características de entrada
                "hidden_layers": model_info["architecture"]["hidden_layers"],  # Configuración de capas ocultas
                "use_batch_norm": model_info["use_batch_norm"],  # Si usa batch normalization
                "activation_fn": model_info["activation_fn"],  # Función de activación
                "dropout_rate": model_info["dropout_rate"],  # Tasa de dropout
                "output_layer_neurons": model_info["architecture"]["output_layer"],  # Neuronas en capa de salida
                "total_parameters": model_info["total_parameters"],  # Total de parámetros entrenables
            },
            "training_configuration": {
                "optimizer": self.optimizer_name,  # Nombre del optimizador
                "learning_rate": self.learning_rate,  # Tasa de aprendizaje
                "weight_decay": self.weight_decay,  # Decaimiento de pesos (L2)
                "loss_function": "BCEWithLogitsLoss",  # Función de pérdida usada
                "use_pos_weight": self.use_pos_weight,  # Si usa peso para clase positiva
                "epochs_run": epochs_run,  # Épocas realmente ejecutadas
                "batch_size": self.batch_size,  # Tamaño del lote
            },
            # Métricas finales redondeadas a 4 decimales, excluyendo NaN
            "final_validation_metrics": {k: round(v, 4) for k, v in final_metrics.items() if not math.isnan(v)}
        }
        
        # Guardar reporte localmente en formato YAML
        report_filename = f"{run_name}_performance_report.yaml"
        report_path = self.local_artifacts_dir / report_filename
        with open(report_path, 'w', encoding='utf-8') as f:
            yaml.dump(report_data, f, indent=2, sort_keys=False)  # indent=2 para legibilidad
        log.info(f"✔ Informe de rendimiento guardado localmente en {report_path}")
        
        # Registrar reporte como artefacto en MLflow
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        log.info("✔ Informe de rendimiento registrado en los artefactos de MLflow.")
        
    def _log_basic_params(self, num_features: int):
        """
        Registra los parámetros básicos del experimento en MLflow.
        
        Args:
            num_features: Número de características de entrada del modelo
        """
        log.info("Usando Metodo _log_basic_params()")
        
        # Registrar parámetros de configuración en MLflow para tracking
        mlflow.log_params({
            # Parámetros de división de datos
            "test_size": self.test_size,  # Proporción de datos para validación
            "random_state": self.random_state,  # Semilla aleatoria
            
            # Parámetros del modelo
            "num_features": num_features,  # Número de características de entrada
            "epochs": self.epochs,  # Número máximo de épocas
            "batch_size": self.batch_size,  # Tamaño del lote
            "hidden_layers": str(self.hidden_layers),  # Configuración de capas ocultas (como string)
            "use_batch_norm": self.use_batch_norm,  # Si usa batch normalization
            "activation_fn": self.activation_fn,  # Función de activación
            "dropout_rate": self.dropout_rate,  # Tasa de dropout
            
            # Parámetros del optimizador
            "optimizer": self.optimizer_name,  # Nombre del optimizador
            "learning_rate": self.learning_rate,  # Tasa de aprendizaje
            "weight_decay": self.weight_decay,  # Decaimiento de pesos
            
            # Parámetros de pérdida y entrenamiento
            "use_pos_weight": self.use_pos_weight,  # Si usa peso para clase positiva
            "scheduler_patience": self.scheduler_patience,  # Paciencia del scheduler
            "early_stopping_patience": self.early_stopping_patience  # Paciencia de early stopping
        })
        
        # Registrar tags adicionales si están configurados en el YAML
        tags = self.params.get("mlflow_config", {}).get("mlflow_tags", [])
        if tags:
            for i, t in enumerate(tags):
                mlflow.set_tag(f"tag_{i}", t)  # Registrar cada tag con un índice
                
    def _log_model_with_signature(self, model: nn.Module, x_example: torch.Tensor):
        """
        Registra el modelo en MLflow con firma (signature) y ejemplo de entrada.
        Intenta usar 'name' (API nueva), si no está disponible hace fallback a 'artifact_path'.
        También añade pip_requirements para manejar dependencias de PyTorch.
        
        Args:
            model: Modelo de PyTorch a registrar
            x_example: Tensor de ejemplo para inferir la firma del modelo
        """
        # Mover modelo a CPU y poner en modo evaluación
        model_cpu = model.to("cpu").eval()
        
        # Generar ejemplo de salida sin calcular gradientes
        with torch.no_grad():
            y_example = model_cpu(x_example).numpy()

        # Inferir la firma del modelo (tipos de entrada y salida) desde ejemplos
        signature = infer_signature(x_example.numpy(), y_example)
        # Guardar ejemplo de entrada para documentación
        input_example = x_example.numpy()

        # Definir dependencias de Python necesarias para usar el modelo
        pip_requirements = [
            # Si usas rueda CUDA local, puedes añadir el índice de PyTorch:
            # "-f https://download.pytorch.org/whl/cu126",
            # "torch==2.7.1+cu126",
            "torch==2.10.0",  # Versión genérica de PyTorch para PyPI
            "scikit-learn",  # Para preprocesamiento
            "pandas",  # Para manipulación de datos
            "numpy",  # Para operaciones numéricas
        ]

        # Suprimir advertencias de PyTorch sobre buffers no escribibles durante la exportación
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*non-writable tensors.*", category=UserWarning)
            try:
                # Intentar usar API nueva de MLflow con 'name' (recomendado)
                mlflow.pytorch.log_model(
                    model_cpu,  # Modelo a registrar
                    name=self.artifact_name_or_path,  # Nombre del artefacto
                    signature=signature,  # Firma de entrada/salida
                    input_example=input_example,  # Ejemplo de entrada
                    pip_requirements=pip_requirements,  # Dependencias necesarias
                    export_model=True  # Exportar modelo completo
                )
            except TypeError:
                # Fallback: usar API antigua con 'artifact_path' para compatibilidad
                mlflow.pytorch.log_model(
                    model_cpu,
                    artifact_path=self.artifact_name_or_path,  # Ruta del artefacto
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements,
                    export_model=True
                )
    
    def _log_plots_and_reports(self, y_true_val: np.ndarray, y_prob_val: np.ndarray):
        """
        Genera gráficos y reportes de evaluación y los registra en MLflow.
        
        Args:
            y_true_val: Etiquetas verdaderas de validación
            y_prob_val: Probabilidades predichas de validación
        """
        # Crear lista de épocas (empezando desde 1)
        epochs = list(range(1, len(self.history["train_loss"]) + 1))

        # ========== GRÁFICOS DE ENTRENAMIENTO VS VALIDACIÓN ==========
        # Gráfico de pérdida (loss) entrenamiento vs validación
        loss_png = self._plot_and_save(epochs, self.history["train_loss"], self.history["val_loss"],
                                    "Training vs Validation Loss", "Loss", "loss_train_val.png")
        # Gráfico de precisión (accuracy) entrenamiento vs validación
        acc_png = self._plot_and_save(epochs, self.history["train_acc"], self.history["val_acc"],
                                    "Training vs Validation Accuracy", "Accuracy", "acc_train_val.png")

        # ========== GRÁFICOS DE EVALUACIÓN EN VALIDACIÓN ==========
        # Curvas ROC y Precision-Recall
        roc_png, pr_png = self._plot_roc_pr(y_true_val, y_prob_val,
                                            roc_file="roc_val.png", pr_file="pr_val.png")
        
        # Convertir probabilidades a predicciones binarias usando umbral 0.5
        y_pred_val = (y_prob_val >= 0.5).astype(int)
        
        # Matriz de confusión
        cm_png = self._plot_confusion_matrix(y_true_val, y_pred_val, filename="confusion_matrix_val.png")

        # ========== REPORTE DE CLASIFICACIÓN ==========
        # Generar reporte de clasificación detallado como texto
        cls_report = classification_report(y_true_val, y_pred_val, target_names=["bad(0)", "good(1)"])
        report_path = self.local_artifacts_dir / "classification_report_val.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(cls_report)

        # ========== REGISTRAR EN MLFLOW ==========
        # Subir todos los gráficos y reportes como artefactos en MLflow
        mlflow.log_artifact(str(loss_png), artifact_path="plots")  # Gráfico de pérdida
        mlflow.log_artifact(str(acc_png), artifact_path="plots")   # Gráfico de precisión
        if roc_png:
            mlflow.log_artifact(str(roc_png), artifact_path="plots")  # Curva ROC (si existe)
        if pr_png:
            mlflow.log_artifact(str(pr_png), artifact_path="plots")    # Curva Precision-Recall (si existe)
        mlflow.log_artifact(str(cm_png), artifact_path="plots")     # Matriz de confusión
        mlflow.log_artifact(str(report_path), artifact_path="reports")  # Reporte de clasificación
        
    def _setup_loss_function(self, y_train: torch.Tensor) -> nn.Module:
        """
        Configura la función de pérdida basándose en los parámetros del YAML.
        Si use_pos_weight está activado, calcula el peso para balancear clases desbalanceadas.
        
        Args:
            y_train: Tensor con las etiquetas de entrenamiento
            
        Returns:
            Función de pérdida configurada (BCEWithLogitsLoss)
        """
        # Si se requiere usar peso para clase positiva (balanceo de clases)
        if self.use_pos_weight:
            # Convertir etiquetas a numpy para contar clases
            y_train_cpu = y_train.detach().cpu().numpy().reshape(-1)
            
            # Contar muestras de cada clase
            pos = float(np.sum(y_train_cpu == 1))  # Clase positiva (buen crédito)
            neg = float(np.sum(y_train_cpu == 0))  # Clase negativa (mal crédito)
            
            # Verificar que ambas clases estén presentes
            if pos == 0 or neg == 0:
                log.warning("Una de las clases no está presente en el batch de entrenamiento. No se usará pos_weight.")
                return nn.BCEWithLogitsLoss()  # Retornar pérdida estándar sin peso
            
            # Calcular peso para clase positiva: proporción de negativos / positivos
            # Esto da más peso a la clase minoritaria durante el entrenamiento
            pos_weight_value = neg / pos
            
            # Crear tensor con el peso en el dispositivo correcto (GPU/CPU)
            pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=self.device)
            
            log.info(f"✔ usando pérdida BCE ponderada con pos_weight={pos_weight_value:.4f}")
            # Retornar pérdida con peso para balancear clases
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            # Usar pérdida estándar sin balanceo
            log.info("✔ usando la pérdida BCE estándar.")
            return nn.BCEWithLogitsLoss()
        
    def train(self):
        """
        Método principal que orquesta todo el pipeline de entrenamiento del modelo.
        Coordina: carga de datos, preprocesamiento, entrenamiento, evaluación y logging.
        """
        log.info(f"✔ hardware usado: {self.device}")
        
        # Configurar experimento de MLflow
        mlflow.set_experiment(self.mlflow_project_name)
        # Obtener prefijo del nombre del run desde configuración o usar default
        run_name_prefix = self.params['mlflow_config'].get('mlflow_run_name_prefix', 'credit_scoring_run')

        # Iniciar un nuevo run de MLflow (contexto que agrupa todas las métricas y artefactos)
        with mlflow.start_run(run_name=f"{run_name_prefix}"):
            log.info("--- Iniciando entrenamiento ---")
            
            # ========== PASO 1: CARGAR Y DIVIDIR DATOS ==========
            df_train, df_val = self._load_and_split_data()
            
            # ========== PASO 2: PREPROCESAR DATOS ==========
            x_train, y_train, x_val, y_val = self._preprocess_data(df_train, df_val)
            num_features = x_train.shape[1]  # Obtener número de características
            
            # Mover datos al dispositivo (GPU si está disponible, sino CPU)
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            x_val, y_val = x_val.to(self.device), y_val.to(self.device)
            
            # ========== PASO 3: CONFIGURAR MODELO, OPTIMIZADOR Y FUNCIÓN DE PÉRDIDA ==========
            log.info(f"✔ inicializando modelo con configuración: {self.hidden_layers}")
            log.info(f"✔ inicializando modelo con {num_features} características de entrada.")
            
            # Crear instancia del modelo con la configuración especificada
            model = CreditScoringModel(
                num_features=num_features,  # Número de características de entrada
                hidden_layers=self.hidden_layers,  # Configuración de capas ocultas
                dropout_rate=self.dropout_rate,  # Tasa de dropout
                use_batch_norm=self.use_batch_norm,  # Si usar batch normalization
                activation_fn=self.activation_fn  # Función de activación
            ).to(self.device)  # Mover modelo al dispositivo
            
            # Configurar función de pérdida
            criterion = self._setup_loss_function(y_train)
            
            # Configurar optimizador según el nombre especificado
            if self.optimizer_name.lower() == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            elif self.optimizer_name.lower() == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            elif self.optimizer_name.lower() == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            else:
                raise ValueError(f"Optimizador {self.optimizer_name} no soportado.")
            log.info(f"✔ usando optimizador: {self.optimizer_name} con lr={self.learning_rate}")
            
            # Configurar scheduler para reducir learning rate cuando no hay mejora
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',  # Reducir cuando la métrica (loss) disminuye
                factor=self.scheduler_factor,  # Factor de reducción del LR
                patience=self.scheduler_patience  # Épocas sin mejora antes de reducir
            )
            
            # Registrar parámetros básicos en MLflow
            self._log_basic_params(num_features=num_features)
            
            # ========== PASO 4: EJECUTAR CICLO DE ENTRENAMIENTO ==========
            epochs_run = self._run_training_loop(model, criterion, optimizer, scheduler, x_train, y_train, x_val, y_val)
            
            # ========== PASO 5: CARGAR MEJOR MODELO Y REGISTRAR ARTEFACTOS ==========
            path_model = f"models/{self.model_name}"
            log.info(f"✔ Cargando el mejor modelo desde {path_model} y registrando artefactos.")
            
            # Cargar los pesos del mejor modelo guardado durante el entrenamiento
            model.load_state_dict(torch.load(path_model, map_location=self.device))
            model.eval()  # Poner modelo en modo evaluación
            
            # Calcular predicciones finales en conjunto de validación
            with torch.no_grad():
                logits_val = model(x_val)  # Obtener logits
                prob_val = torch.sigmoid(logits_val).detach().cpu().numpy().reshape(-1)  # Convertir a probabilidades
                y_val_np = y_val.detach().cpu().numpy().reshape(-1)  # Convertir etiquetas a numpy
                
            # Calcular métricas finales
            final_metrics = self._compute_metrics(y_val_np, prob_val, threshold=0.5)
            # Registrar métricas finales en MLflow con prefijo "final_val_"
            mlflow.log_metrics({f"final_val_{k}": v for k, v in final_metrics.items() if not math.isnan(v)})

            # ========== PASO 6: GENERAR Y REGISTRAR GRÁFICOS Y REPORTES ==========
            self._log_plots_and_reports(y_val_np, prob_val)
            
            # ========== PASO 7: REGISTRAR MODELO Y PREPROCESADOR EN MLFLOW ==========
            # Generar y registrar reporte de rendimiento
            self._generate_and_log_performance_report(model, final_metrics, num_features, epochs_run, run_name_prefix)
            
            # Preparar ejemplo de entrada para la firma del modelo (primeras 5 muestras)
            x_example = x_train[:5].detach().cpu()
            # Registrar modelo con firma y ejemplo de entrada
            self._log_model_with_signature(model, x_example)
            
            # Registrar preprocesador como artefacto
            path_preprocessor = f"models/{self.preprocessor_filename}"
            mlflow.log_artifact(path_preprocessor, artifact_path="preprocessing")
            log.info("✔ Preprocesador y modelo guardados en MLflow.")
            

# ========== PUNTO DE ENTRADA PRINCIPAL ==========
if __name__ == "__main__":
    """
    Punto de entrada principal cuando se ejecuta el script directamente.
    Configura logging, parsea argumentos de línea de comandos y ejecuta el entrenamiento.
    """
    # Configurar sistema de logging con nivel INFO por defecto
    setup_logging()
    
    # Crear parser de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Optimizador de umbral para el modelo de clasificación de ID.")
    
    # Agregar argumento para ruta del archivo de configuración
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/credit_scoring-training_config-german_credit_risk_v110.yaml",
        help="Ruta al archivo de configuración YAML del optimizador de umbral."
    )
    
    # Parsear argumentos de línea de comandos
    cli_args = parser.parse_args()
    log.info(f"Configurar ruta: {cli_args.config}")
    
    # Ejecutar entrenamiento con manejo de errores
    try:
        # Crear instancia del entrenador con la configuración especificada
        trainer = CreditScoringModelTraining(Path(cli_args.config))
        # Ejecutar el pipeline de entrenamiento completo
        trainer.train()
    except Exception as e:
        # Registrar error completo con stack trace si ocurre alguna excepción
        log.error(f"Error al ejecutar el entrenamiento: {e}", exc_info=True)
        
"""
Ejemplos de ejecución del archivo de entrenamiento:
python src/training/train.py --config config/training/credit_scoring-training_config-german_credit_risk_v100.yaml
python src/training/train.py --config config/training/credit_scoring-training_config-german_credit_risk_v110.yaml
python src/training/train.py --config config/training/credit_scoring-training_config-german_credit_risk_v120.yaml
python src/training/train.py --config config/training/credit_scoring-training_config-german_credit_risk_v130.yaml
"""