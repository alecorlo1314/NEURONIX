# 🪙 🏦 Credit Scoring Inference API

Este microservicio proporciona una API RESTful para predecir el riesgo crediticio de un solicitante utilizando una red neuronal Perceptrón Multicapa (MLP) construida con PyTorch.

Este proyecto es parte del curso de Deep Learning de la plataforma [inGeniia.co](https://www.ingeniia.co), donde enseñamos a llevar modelos desde la teoría hasta la producción en la nube.

---

## 📋 Descripción
El servicio recibe datos demográficos y financieros de un solicitante, los procesa utilizando un pipeline de transformación (scikit-learn) y los pasa a través de un modelo MLP entrenado para clasificación binaria.

### Características Principales

- API Rápida: Construida sobre FastAPI para alto rendimiento.

- Validación de Tipos: Uso de Pydantic para garantizar que los datos de entrada cumplan con los esquemas requeridos (validación de enums para sexo, vivienda, propósito, etc.).

- Inferencia Deep Learning: Modelo MLP (Multi-Layer Perceptron) optimizado con capas ocultas, Dropout y Batch Normalization.

- Arquitectura Modular: Separación clara entre la lógica del servidor (server) y la lógica de inferencia (inference).

- Container-Ready: Dockerizado y optimizado para despliegue en Google Cloud Run.

---

## 🚀 Guía de Ejecución

### Paso 1: Preparación de Artefactos
- Asegúrate de tener los artefactos del modelo (`.pt`) y el preprocesador (`.joblib`) en la carpeta `Python/puntaje_credito/models/`.

### Paso 2: Construcción de la Imagen Docker
- Navega al directorio raíz `NEURONIX/` y ejecuta el siguiente comando para construir la imagen.

```bash
docker build -t neuronix/puntaje-credito-mlp:1.0 -f container-images/puntaje_credito_service/Dockerfile .
```
### Paso 3: Ejecutar el Contenedor Docker
- Una vez construida la imagen, levanta un contenedor con este comando:

```bash
 docker run -d -p 8000:8000 --name puntaje-credito-service neuronix/puntaje-credito-mlp:1.0
```

### Paso 4: Verificar el Funcionamiento
- Abre tu navegador web y ve a la siguiente URL para acceder a la documentación interactiva de la API:

```bash
http://localhost:8000/docs
```

---

## 📝 Cómo Usar la API (¡Haciendo una Predicción!)
El endpoint principal es /mlp_demo. Puedes enviarle una solicitud POST con los datos del solicitante en formato JSON.

- Opción A: Usando la Documentación Interactiva (Swagger)

    - Ve a http://localhost:8000/docs.

    - Despliega el endpoint POST /mlp_demo.

    - Haz clic en el botón "Try it out".

    - Modifica el cuerpo de la solicitud (Request body) con los datos del cliente.

    - Haz clic en "Execute". ¡Verás la respuesta del modelo directamente en la página!

- Opción B: Usando cURL desde la Terminal

    - Abre una terminal y ejecuta el siguiente comando cURL para enviar una solicitud de ejemplo:

        ```bash
        curl -X 'POST' \
        'http://localhost:8000/mlp_demo' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "Age": 35,
        "Sex": "male",
        "Job": 2,
        "Housing": "own",
        "Saving accounts": "little",
        "Checking account": "moderate",
        "Credit amount": 2500,
        "Duration": 24,
        "Purpose": "car"
        }'
        ```

- Respuesta Exitosa Esperada (200 OK):
Si todo va bien, recibirás una respuesta como esta, indicando la predicción (good o bad) y la probabilidad asociada:

    ```bash
    {
    "prediction": "good",
    "probability": 0.7852
    }
    ```

---

## ⚙️ Gestión del Contenedor
Aquí tienes algunos comandos útiles para administrar el contenedor Docker.

- Puedes detener el contenedor en cualquier momento usando:

    ```bash
    docker stop credit-scoring-service  
    ```

- Ver los logs en tiempo real:
    ```bash
    docker logs -f credit-scoring-service 
    ```

- Reiniciar un contenedor detenido::
    ```bash
    docker start credit-scoring-service 
    ```

- Reiniciar un contenedor detenido::
    ```bash
    docker stop credit-scoring-service && docker rm credit-scoring-service
    ```

---

## ☁️ Despliegue en Google Cloud Platform (GCP)

Este servicio está diseñado para una arquitectura Serverless utilizando Cloud Run. El flujo de CI/CD se maneja mediante Cloud Build.

### Flujo de Despliegue (Cloud Build)

El archivo `ops/cloudbuild-credit_scoring_service.yaml` gestiona los siguientes pasos automáticamente:

1. **Build:** Construye la imagen Docker utilizando el `Dockerfile` optimizado (multi-stage).
2. **Push:** Sube la imagen a `Artifact Registry` (us-central1-docker.pkg.dev/...).
3. **Deploy:** Despliega la nueva imagen en Cloud Run como un servicio gestionado.

### Configuración del Despliegue

- **Región:** `us-central1`
- **Autenticación:** `--allow-unauthenticated`
- **Memoria:** `1Gi`
- **Puerto:** `8080`

---

## 📄 Licencia y Atribución
Este proyecto se distribuye bajo la Licencia MIT, pero con una cláusula adicional de atribución educativa.
**Copyright (c) 2024 inGeniia.co**

Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia de este software y los archivos de documentación asociados, para tratar el Software sin restricciones, incluido el uso, copia, modificación, fusión, publicación y distribución.

### Condición de uso: 
Si utilizas este código, modelos o arquitectura en tu propio proyecto, investigación o producto comercial, debes incluir explícitamente la siguiente mención en tu documentación, README o sección de "Acerca de":

```text
Este software implementa arquitecturas de Deep Learning basadas en los materiales educativos de inGeniia.co. 
El modelo original de Credit Scoring fue desarrollado por el equipo de inGeniia.
```

---

**¿Te interesa aprender más?** 

Visita [www.inGeniia.co](https://www.ingeniia.co) para acceder al código fuente completo, los videos explicativos y los mapas mentales de esta y otras redes neuronales.


