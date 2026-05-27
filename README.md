# 📊 Aplicación de Deep Learning para Clasificación de Riesgo Financiero

Este proyecto contiene una aplicación interactiva desarrollada en **Streamlit** que integra modelos avanzados de Machine Learning y Deep Learning para la evaluación y clasificación del riesgo de crédito de clientes. 

La plataforma web combina un backend analítico que procesa datos estructurados con una interfaz visual intuitiva para el análisis exploratorio y la predicción en tiempo real.

---

## 🎯 Objetivo del Proyecto

El objetivo principal es construir y desplegar un pipeline robusto e interactivo de extremo a extremo (*End-to-End*) que automatice la ingesta de datos de clientes, realice un análisis exploratorio avanzado y evalúe el riesgo financiero utilizando arquitecturas de redes neuronales y modelos de ensamble. 

El proyecto destaca por su naturaleza híbrida, utilizando **R** para tareas específicas de extracción y transformación de datos, y **Python** como motor principal para el modelado predictivo y la interfaz de usuario.

---

## 🧩 Problema que Resuelve la Aplicación

En el sector financiero, evaluar la concesión de un crédito de forma manual consume una cantidad ingente de tiempo y está sujeto a sesgos humanos. Esta aplicación resuelve el problema mediante la **automatización de la clasificación del riesgo (Clientes *Good* vs. *Bad*)**:

1. **Elimina el sesgo de volumen:** Permite a los analistas evaluar visualmente las tasas de morosidad relativas por categorías mediante gráficos de proporciones apiladas al 100%.
2. **Mitiga pérdidas financieras:** Optimiza los modelos bajo funciones de coste personalizadas que penalizan con mayor severidad los falsos negativos (aprobar un crédito a un cliente con perfil de alto riesgo).
3. **Decisiones reproducibles:** Ofrece una sección de predicción donde se introducen los parámetros de un cliente y se genera un diagnóstico estandarizado e inmediato basado en el modelo entrenado.

---

## 💻 Requisitos del Sistema

Para garantizar la correcta ejecución del pipeline híbrido, el sistema anfitrión debe contar con:
* **Python 3.11** o **Python 3.12** (Entorno optimizado)
* **R-base** instalado en el sistema (el ejecutable `Rscript` debe estar accesible en el PATH del sistema o en `/usr/bin/Rscript`).

---

## 🛠️ Creación del Entorno

Para evitar conflictos de dependencias en local, se recomienda crear un entorno virtual limpio utilizando el gestor rápido `uv` o el módulo nativo `venv`.

### Opción A: Usando `uv` (Recomendado por su velocidad)
Si no tienes `uv` instalado, puedes instalarlo mediante pip, y luego crear el entorno:
```bash
pip install uv
uv venv --python 3.11