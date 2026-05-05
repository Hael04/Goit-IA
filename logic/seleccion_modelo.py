# --- seleccion_modelo.py ---
import time
from models.modelo_knn import obtener_respuesta_knn
from models.modelo_llm import obtener_cadena_rag

# Tiempo mínimo entre reintentos de conexión al LLM (segundos)
_MIN_SEGUNDOS_REINTENTO_LLM = 60


class SelectorDeModelo:
    def __init__(self, usar_knn=True, usar_llm=True, umbral_distancia=0.2):
        self.usar_knn = usar_knn
        self.usar_llm = usar_llm
        self.UMBRAL_DISTANCIA_COSINE = umbral_distancia
        self.rag_chain = None

        # Control de reintentos: 0.0 → el primer intento se hace en la primera consulta
        self._ultimo_intento_llm = 0.0

    # ──────────────────────────────────────────────────────────
    # Inicialización diferida del LLM
    # ──────────────────────────────────────────────────────────

    def _init_llm_si_necesario(self):
        """
        Conecta con Chroma Cloud y crea la cadena RAG de forma diferida.
        Solo reintenta si han pasado al menos _MIN_SEGUNDOS_REINTENTO_LLM
        desde el último intento fallido, evitando bloquear cada petición.
        """
        if self.rag_chain is not None:
            return   # ya está listo

        if not self.usar_llm:
            return   # desactivado permanentemente por configuración

        segundos_desde_ultimo = time.monotonic() - self._ultimo_intento_llm
        if segundos_desde_ultimo < _MIN_SEGUNDOS_REINTENTO_LLM:
            return   # aún en período de espera tras un fallo anterior

        self._ultimo_intento_llm = time.monotonic()

        try:
            print("🔄 Intentando conectar con el LLM (RAG)...")
            cadena = obtener_cadena_rag()
            if cadena:
                self.rag_chain = cadena
                print("✅ Modelo LLM listo.")
            else:
                print("⚠️ LLM: colección Chroma no encontrada. Entrena el modelo desde el panel admin.")
        except Exception as e:
            print(f"❌ Error conectando con LLM: {e}. Se reintentará en {_MIN_SEGUNDOS_REINTENTO_LLM}s.")

    # ──────────────────────────────────────────────────────────
    # Lógica principal de respuesta
    # ──────────────────────────────────────────────────────────

    def responder(self, pregunta, historial="", forzar_llm=False):
        """
        Lógica híbrida de selección de modelo:

        1. KNN: busca coincidencia semántica en el caché FAQ (inicialización diferida interna).
           - FAQ BLOQUEADA + distancia aceptable → siempre devuelve esa respuesta (ignora forzar_llm).
           - FAQ normal + distancia aceptable + no forzar_llm → devuelve desde caché.
        2. LLM RAG: se usa cuando KNN no aplica o se fuerza regeneración (inicialización diferida aquí).

        Retorna: (respuesta: str, fuente: str, bloqueado: bool)
        """

        # 1. Intentar KNN (el módulo gestiona su propia inicialización diferida)
        if self.usar_knn:
            respuesta_knn, distancia, bloqueado = obtener_respuesta_knn(pregunta)

            if respuesta_knn and distancia <= self.UMBRAL_DISTANCIA_COSINE:
                if bloqueado:
                    print(f"[Selector] FAQ BLOQUEADA activada (distancia={distancia:.4f})")
                    return respuesta_knn, "KNN (Bloqueado)", True

                if not forzar_llm:
                    print(f"[Selector] KNN caché activado (distancia={distancia:.4f})")
                    return respuesta_knn, "KNN (Caché Semántico)", False

        # 2. LLM RAG — inicializar si aún no está listo
        self._init_llm_si_necesario()

        if self.rag_chain:
            try:
                respuesta_llm = self.rag_chain.invoke({
                    "question": pregunta,
                    "history":  historial
                })
                return respuesta_llm, "LLM (RAG Generativo)", False
            except Exception as e:
                print(f"[Selector] Error en RAG: {e}. Invalidando cadena para forzar reconexión.")
                # Invalida la cadena para que el siguiente intento reconecte
                self.rag_chain = None
                self._ultimo_intento_llm = 0.0
                return "Ocurrió un error al generar la respuesta. Por favor, intenta de nuevo en unos momentos.", "Error", False

        return (
            "El sistema de respuestas no está disponible en este momento. "
            "Por favor, intenta de nuevo en unos minutos.",
            "Nulo",
            False
        )
