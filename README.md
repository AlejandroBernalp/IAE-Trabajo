# 🤖 Proyecto de Inteligencia Artificial - Clasificación de Crédito

Este repositorio contiene el flujo de trabajo para el análisis y modelado de datos de crédito utilizando Python y R.

## 📋 Estructura del Proyecto

Actualmente, el pipeline se divide en dos fases principales:

### 1. Preprocesamiento y Limpieza (R)
*   **Archivo:** `import_y_limpieza.R`
*   **Descripción:** 
    *   Conecta con el repositorio UCI mediante el paquete `ucimlrepo`.
    *   **Tratamiento de datos:** Traduce los códigos técnicos (ej. *A11, A12*) y nombres genéricos de columnas (*Attribute1, Attribute2*) a términos comprensibles y descriptivos.
    *   **Salida:** Genera un archivo local `german_credit_clean.csv` (ignorado en Git por su peso, pero necesario para el siguiente paso).

### 2. Ingeniería de Características (Python)
*   **Archivo:** `main.py`
*   **Descripción:** 
    *   Carga el dataset limpio generado en el paso anterior.
    *   **Codificación:** Aplica técnicas de transformación para preparar los datos para modelos de ML:
        *   **One-Hot Encoding** para variables nominales.
        *   **Binary Encoding** para variables dicotómicas.
        *   **Ordinal Encoding** para variables con jerarquía.
*   **Próximamente:** Implementación y evaluación de modelos de Machine Learning.

---

## 🛠️ Instalación y Configuración

### Requisitos de Python (uv)
Este proyecto utiliza `uv` para la gestión de dependencias. Para sincronizar tu entorno virtual:
```bash
uv sync