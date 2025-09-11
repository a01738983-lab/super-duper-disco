# 📊 Dashboard Interactivo de Análisis de Regresión

Este proyecto consiste en un **dashboard interactivo desarrollado en Python con Streamlit**, diseñado para analizar datos de producción (ejemplo: velocidad de proceso y factor de merma) a partir de variables de materia prima.  

El sistema permite:  
- 📥 Cargar archivos Excel con datos de producción.  
- 🧹 Limpiar y procesar la información automáticamente.  
- 🤖 Entrenar modelos de **regresión lineal múltiple**.  
- 📈 Generar métricas clave (R², RMSE, MAE).  
- 📊 Visualizar correlaciones, distribuciones y residuos.  
- 📤 Exportar resultados con predicciones a Excel.  

---

## 🚀 Demo en línea

Puedes acceder al dashboard sin necesidad de instalar nada desde este enlace:  

👉 [Abrir Dashboard en Streamlit](https://reto-equipo1-leonali.streamlit.app/)  

---

## 💻 Repositorio en GitHub

El código fuente de este proyecto está disponible aquí:  

👉 [Ir al repositorio](https://github.com/a01738983-lab/super-duper-disco)  

---

## ⚙️ Requisitos de instalación (para correrlo localmente)

1. Clonar este repositorio:  
   ```bash
   git clone https://github.com/a01738983-lab/super-duper-disco.git
   cd super-duper-disco


2. Instalar las dependencias:
   pip install -r requirements.txt


3. Ejecutar la aplicación:
   streamlit run app.py



📊 Uso
- Subir un archivo Excel con las siguientes características:
- Variables predictoras: deben contener _MP en el nombre.
- Variables objetivo: deben incluir vel, fm o merma en el nombre.
- Columnas opcionales: MateriaPrima, Proveedor.
- Filtrar por materia prima y proveedores según se requiera.
- Seleccionar variables objetivo a analizar.
- Revisar las métricas, gráficas y conclusiones automáticas.
- Descargar el archivo Excel con predicciones generadas por los modelos.
