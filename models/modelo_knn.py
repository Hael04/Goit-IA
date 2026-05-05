# models/modelo_knn.py
import os
import sys
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS DE BASE DE DATOS (MongoDB) ---
from database import faq_collection

# --- MODELO DE EMBEDDINGS VÍA API ---
HF_TOKEN = os.getenv("HF_TOKEN")

print("🔄 Conectando con API de embeddings (HuggingFace)...")
modelo_embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    huggingfacehub_api_token=HF_TOKEN
)
print("✅ API de embeddings lista.")

# --- VARIABLES GLOBALES DEL MODELO KNN ---
knn_model       = None
respuestas_knn  = []
bloqueado_flags = []

# Control de reintentos: evita llamadas repetidas al arrancar
_ultimo_intento  = 0.0
_MIN_SEGUNDOS_REINTENTO = 30   # no reintentar más frecuente que cada 30 s


def inicializar_knn():
    """
    Carga los datos desde MongoDB y entrena el modelo KNN.
    Puede llamarse al arrancar y también de forma diferida desde
    obtener_respuesta_knn() si el arranque falló.
    """
    global knn_model, respuestas_knn, bloqueado_flags, _ultimo_intento

    _ultimo_intento = time.monotonic()

    try:
        print("🔄 Cargando base de conocimiento FAQ desde MongoDB...")

        documentos = list(faq_collection.find(
            {}, {"_id": 0, "pregunta": 1, "respuesta": 1, "bloqueado": 1}
        ))

        if not documentos:
            print("⚠️ La colección FAQ está vacía. KNN desactivado hasta que haya FAQs.")
            knn_model       = None
            respuestas_knn  = []
            bloqueado_flags = []
            return

        preguntas       = [doc['pregunta'] for doc in documentos]
        respuestas_knn  = [doc['respuesta'] for doc in documentos]
        bloqueado_flags = [doc.get('bloqueado', False) for doc in documentos]

        # Generar embeddings de todas las preguntas vía API
        X_dataset = np.array(modelo_embedding.embed_documents(preguntas))

        # n_neighbors=min(3, total): top-3 candidatos para mayor robustez
        n_vecinos = min(3, len(preguntas))
        knn_model = NearestNeighbors(n_neighbors=n_vecinos, metric='cosine')
        knn_model.fit(X_dataset)

        bloqueadas = sum(1 for b in bloqueado_flags if b)
        print(
            f"✅ Modelo KNN listo. Total: {len(respuestas_knn)} FAQs "
            f"({bloqueadas} bloqueadas, {n_vecinos} vecinos activos)."
        )

    except Exception as e:
        print(f"⚠️ No se pudo inicializar KNN. Usando solo LLM. Detalle: {e}")
        knn_model = None


# --- Intento de carga al arrancar (puede fallar si la red no está lista) ---
try:
    inicializar_knn()
except Exception as e:
    print(f"⚠️ KNN: fallo silencioso en el arranque ({e}). Se reintentará en la primera consulta.")
    knn_model = None


def obtener_respuesta_knn(pregunta_usuario):
    """
    Busca la FAQ más similar usando embeddings semánticos.

    Si el modelo no está listo (falló al arrancar), intenta inicializarlo
    de forma diferida antes de responder. El reintento respeta un intervalo
    mínimo para no bloquear cada petición.

    Retorna (respuesta, distancia_coseno, bloqueado):
    - respuesta    : texto de la FAQ más cercana, o None.
    - distancia    : 0.0 = idéntico, 1.0 = completamente diferente.
    - bloqueado    : True si la FAQ tiene respuesta fija e inamovible.
    """
    global knn_model, _ultimo_intento

    # Inicialización diferida: si falló al arrancar, reintenta ahora
    if knn_model is None:
        segundos_desde_ultimo = time.monotonic() - _ultimo_intento
        if segundos_desde_ultimo >= _MIN_SEGUNDOS_REINTENTO:
            print("[KNN] Reintentando inicialización diferida...")
            inicializar_knn()

    if knn_model is None:
        return None, 1.0, False

    try:
        X_usuario = np.array(
            modelo_embedding.embed_query(pregunta_usuario)
        ).reshape(1, -1)

        distancias, indices = knn_model.kneighbors(X_usuario)

        indice_mejor   = indices[0][0]
        distancia_mejor = distancias[0][0]

        respuesta = respuestas_knn[indice_mejor]
        bloqueado = bloqueado_flags[indice_mejor] if bloqueado_flags else False

        print(f"[KNN] Distancia coseno: {distancia_mejor:.4f} | Bloqueado: {bloqueado}")

        return respuesta, distancia_mejor, bloqueado

    except Exception as e:
        print(f"[KNN] Error en predicción: {e}")
        return None, 1.0, False
