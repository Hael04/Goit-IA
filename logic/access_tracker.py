from datetime import datetime
from collections import Counter
import sys
import os

# --- CONFIGURACIÓN DE RUTAS ---
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS DE BASE DE DATOS (MongoDB) ---
from database import (
    insert_access_log, get_all_access_logs,
    insert_chat_log,
)


# ──────────────────────────────────────────────────────────────
# REGISTRO DE ACCESO  (paso del modal: matrícula + programa)
# ──────────────────────────────────────────────────────────────

def registrar_acceso(programa: str, ip: str,
                     dispositivo: str, matricula: str = "") -> bool:
    ahora = datetime.now()
    try:
        insert_access_log(
            dia        = ahora.strftime('%A'),
            fecha      = ahora.strftime('%Y-%m-%d'),
            hora       = ahora.strftime('%H:%M:%S'),
            programa   = programa,
            dispositivo= dispositivo,
            ip         = ip,
            matricula  = matricula,
        )
        print(f"✅ Acceso registrado: {matricula} | {programa} desde {ip}")
        return True
    except Exception as e:
        print(f"❌ Error registrando acceso en MongoDB: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# REGISTRO DE PREGUNTA
# ──────────────────────────────────────────────────────────────

def registrar_pregunta(matricula: str, programa: str,
                       pregunta: str, respuesta: str,
                       modelo: str) -> bool:
    ahora = datetime.now()
    try:
        insert_chat_log(
            matricula = matricula,
            programa  = programa,
            pregunta  = pregunta,
            respuesta = respuesta,
            modelo    = modelo,
            fecha     = ahora.strftime('%Y-%m-%d'),
            hora      = ahora.strftime('%H:%M:%S'),
        )
        print(f"✅ Pregunta registrada: {matricula} | modelo={modelo}")
        return True
    except Exception as e:
        print(f"❌ Error registrando pregunta en MongoDB: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# ESTADÍSTICAS Y LISTADOS  (panel admin — sin pandas)
# ──────────────────────────────────────────────────────────────

def obtener_estadisticas_diarias() -> dict:
    """Conteo de accesos por programa usando collections.Counter."""
    try:
        registros = get_all_access_logs()
        if not registros:
            return {}
        return dict(Counter(r.get('programa', '') for r in registros if r.get('programa')))
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas: {e}")
        return {}


def obtener_todos_los_registros() -> list[dict]:
    """Registros de acceso ordenados del más reciente al más antiguo."""
    try:
        registros = get_all_access_logs()
        if not registros:
            return []

        # Garantizar campo matricula en registros antiguos
        for r in registros:
            r.setdefault('matricula', '')
            # Reemplazar None por cadena vacía en todos los campos
            for k, v in r.items():
                if v is None:
                    r[k] = ''

        # Ordenar por fecha y hora descendente (MongoDB ya los entrega así,
        # pero lo reforzamos en caso de registros sin índice)
        registros.sort(key=lambda r: (r.get('fecha', ''), r.get('hora', '')), reverse=True)
        return registros

    except Exception as e:
        print(f"❌ Error obteniendo registros históricos: {e}")
        return []
