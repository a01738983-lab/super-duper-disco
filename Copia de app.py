# app.py - Dashboard Interactivo para Análisis de Velocidades de Espinaca

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# Eliminado RandomForestRegressor - solo usando regresión lineal
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Análisis de Regresión",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Carga y limpieza de datos ---
@st.cache_data
def load_data(path):
    """Carga los datos del archivo Excel (función cacheada sin widgets)"""
    try:
        df = pd.read_excel(path)
        return df, None
    except Exception as e:
        return None, str(e)

def process_and_clean_data(df, materia_seleccionada, proveedores_excluir, selected_targets):
    """Procesa y limpia los datos según las selecciones del usuario"""
    # Filtrar registros (opcional)
    df_filtered = df.copy()
    
    # Filtrar por materia prima
    if 'MateriaPrima' in df.columns and materia_seleccionada != 'Todas':
        df_filtered = df_filtered[df_filtered['MateriaPrima'] == materia_seleccionada]
    
    # Excluir proveedores
    if 'Proveedor' in df.columns and proveedores_excluir:
        df_filtered = df_filtered[~df_filtered['Proveedor'].isin(proveedores_excluir)]
    
    # Asignar variables objetivo
    vel_col = selected_targets[0] if len(selected_targets) > 0 else None
    fm_col = selected_targets[1] if len(selected_targets) > 1 else None
    
    # Eliminar filas con nulos en variables objetivo
    if selected_targets:
        df_clean = df_filtered.dropna(subset=selected_targets)
    else:
        df_clean = df_filtered.copy()
    
    # Columnas numéricas con "_MP" (excluyendo Fecha_entrada_MP)
    mp_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns 
               if '_mp' in col.lower() and 'fecha_entrada_mp' not in col.lower()]
    
    # Imputar valores faltantes en predictoras numéricas con mediana
    for col in mp_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    return df_clean, mp_cols, vel_col, fm_col

def show_data_interface(df):
    """Muestra la interfaz de usuario para seleccionar opciones de filtrado"""
    st.success(f"✅ Archivo cargado exitosamente: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # Mostrar información inicial
    st.write("**Datos originales:**")
    st.write(f"- Registros totales: {len(df)}")
    
    # Verificar columnas necesarias
    required_cols = ['MateriaPrima', 'Proveedor']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Columnas faltantes: {missing_cols}")
        st.write("Columnas disponibles:", list(df.columns))
    
    # Selección de materia prima
    materia_seleccionada = 'Todas'
    if 'MateriaPrima' in df.columns:
        materias_disponibles = df['MateriaPrima'].unique()
        st.write(f"- Materias primas disponibles: {', '.join(materias_disponibles)}")
        
        if len(materias_disponibles) > 1:
            materia_seleccionada = st.selectbox(
                "Selecciona la materia prima a analizar:",
                ['Todas'] + list(materias_disponibles)
            )
    else:
        st.info("ℹ️ Columna 'MateriaPrima' no encontrada, usando todos los datos")
    
    # Selección de proveedores a excluir
    proveedores_excluir = []
    if 'Proveedor' in df.columns:
        df_temp = df.copy()
        if materia_seleccionada != 'Todas':
            df_temp = df_temp[df_temp['MateriaPrima'] == materia_seleccionada]
        
        proveedores_disponibles = df_temp['Proveedor'].unique()
        st.write(f"- Proveedores disponibles: {len(proveedores_disponibles)}")
        
        proveedores_excluir = st.multiselect(
            "Selecciona proveedores a excluir (opcional):",
            proveedores_disponibles
        )
    else:
        st.info("ℹ️ Columna 'Proveedor' no encontrada")
    
    # Buscar variables objetivo
    df_temp = df.copy()
    if materia_seleccionada != 'Todas' and 'MateriaPrima' in df.columns:
        df_temp = df_temp[df_temp['MateriaPrima'] == materia_seleccionada]
    if proveedores_excluir and 'Proveedor' in df.columns:
        df_temp = df_temp[~df_temp['Proveedor'].isin(proveedores_excluir)]
    
    # Buscar automáticamente variables objetivo comunes
    vel_cols = [col for col in df_temp.columns if 'vel' in col.lower()]
    fm_cols = [col for col in df_temp.columns if ('fm' in col.lower() or 'merma' in col.lower()) and 'fmi' not in col.lower()]
    
    # También buscar otras variables numéricas que podrían ser objetivos
    numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
    potential_targets = [col for col in numeric_cols if '_mp' not in col.lower() and 'fecha' not in col.lower()]
    
    # Combinar todas las posibles variables objetivo
    all_potential_targets = list(set(vel_cols + fm_cols + potential_targets))
    
    selected_targets = []
    if all_potential_targets:
        st.write(f"- Variables objetivo potenciales encontradas: {len(all_potential_targets)}")
        
        selected_targets = st.multiselect(
            "Selecciona las variables objetivo a analizar:",
            all_potential_targets,
            default=all_potential_targets[:2] if len(all_potential_targets) >= 2 else all_potential_targets
        )
    else:
        st.error("❌ No se encontraron variables objetivo válidas")
    
    return materia_seleccionada, proveedores_excluir, selected_targets

# --- 2. Variables objetivo y predictoras ---
def get_features_targets(df, mp_cols, vel_col, fm_col):
    """Extrae características y variables objetivo"""
    X = df[mp_cols]
    y_vel = df[vel_col] if vel_col else None
    y_fm = df[fm_col] if fm_col else None
    return X, y_vel, y_fm

# --- 3. Modelado ---
def train_models(X, y, model_name):
    """Entrena modelo de regresión lineal"""
    if y is None or len(y) == 0:
        return None, None, None, None, None, None, None, None, None
    
    # División 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Regresión Lineal
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_train_pred_lr = lr.predict(X_train_scaled)
    y_test_pred_lr = lr.predict(X_test_scaled)
    
    # Métricas Regresión Lineal
    metrics_lr = {
        'r2_train': r2_score(y_train, y_train_pred_lr),
        'r2_test': r2_score(y_test, y_test_pred_lr),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred_lr)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred_lr)),
        'mae_train': mean_absolute_error(y_train, y_train_pred_lr),
        'mae_test': mean_absolute_error(y_test, y_test_pred_lr),
    }
    
    # Residuos
    residuals_lr = y_test - y_test_pred_lr
    
    return (lr, scaler, X_train, X_test, y_train, y_test, 
            y_test_pred_lr, residuals_lr, metrics_lr)

# --- 4. Análisis de importancia ---
def get_feature_importance(lr_model, feature_names, scaler):
    """Obtiene importancia de características para regresión lineal"""
    # Coeficientes de Regresión Lineal (valores absolutos)
    lr_importance = pd.Series(np.abs(lr_model.coef_), index=feature_names).sort_values(ascending=False)
    
    return lr_importance

def train_pareto_models(X, y_vel, y_fm, mp_cols, scaler_vel, scaler_fm, lr_vel, lr_fm, top_n=6):
    """
    Entrena modelos de regresión lineal usando solo las top N variables más importantes del análisis de Pareto.
    
    Args:
        X: DataFrame con las variables predictoras
        y_vel: Serie con la variable objetivo de velocidad
        y_fm: Serie con la variable objetivo de factor de merma
        mp_cols: Lista de nombres de columnas MP
        scaler_vel: Scaler entrenado para velocidad
        scaler_fm: Scaler entrenado para factor de merma
        lr_vel: Modelo de regresión lineal entrenado para velocidad
        lr_fm: Modelo de regresión lineal entrenado para factor de merma
        top_n: Número de variables más importantes a seleccionar (default: 6)
    
    Returns:
        dict: Diccionario con modelos, métricas y variables seleccionadas
    """
    results = {}
    
    # Obtener importancias para velocidad
    if y_vel is not None and lr_vel is not None:
        importance_vel = get_feature_importance(lr_vel, mp_cols, scaler_vel)
        top_vars_vel = importance_vel.head(top_n).index.tolist()
        
        # Entrenar modelo con variables seleccionadas
        X_selected_vel = X[top_vars_vel]
        X_train_vel, X_test_vel, y_train_vel, y_test_vel = train_test_split(
            X_selected_vel, y_vel, test_size=0.2, random_state=42
        )
        
        # Escalar datos
        scaler_pareto_vel = StandardScaler()
        X_train_scaled_vel = scaler_pareto_vel.fit_transform(X_train_vel)
        X_test_scaled_vel = scaler_pareto_vel.transform(X_test_vel)
        
        # Entrenar modelo
        lr_pareto_vel = LinearRegression()
        lr_pareto_vel.fit(X_train_scaled_vel, y_train_vel)
        
        # Predicciones
        y_pred_pareto_vel = lr_pareto_vel.predict(X_test_scaled_vel)
        
        # Métricas
        metrics_pareto_vel = {
            'r2_test': r2_score(y_test_vel, y_pred_pareto_vel),
            'rmse_test': np.sqrt(mean_squared_error(y_test_vel, y_pred_pareto_vel)),
            'mae_test': mean_absolute_error(y_test_vel, y_pred_pareto_vel)
        }
        
        results['velocidad'] = {
            'model': lr_pareto_vel,
            'scaler': scaler_pareto_vel,
            'top_variables': top_vars_vel,
            'metrics': metrics_pareto_vel,
            'y_test': y_test_vel,
            'y_pred': y_pred_pareto_vel,
            'X_test': X_test_vel
        }
    
    # Obtener importancias para factor de merma
    if y_fm is not None and lr_fm is not None:
        importance_fm = get_feature_importance(lr_fm, mp_cols, scaler_fm)
        top_vars_fm = importance_fm.head(top_n).index.tolist()
        
        # Entrenar modelo con variables seleccionadas
        X_selected_fm = X[top_vars_fm]
        X_train_fm, X_test_fm, y_train_fm, y_test_fm = train_test_split(
            X_selected_fm, y_fm, test_size=0.2, random_state=42
        )
        
        # Escalar datos
        scaler_pareto_fm = StandardScaler()
        X_train_scaled_fm = scaler_pareto_fm.fit_transform(X_train_fm)
        X_test_scaled_fm = scaler_pareto_fm.transform(X_test_fm)
        
        # Entrenar modelo
        lr_pareto_fm = LinearRegression()
        lr_pareto_fm.fit(X_train_scaled_fm, y_train_fm)
        
        # Predicciones
        y_pred_pareto_fm = lr_pareto_fm.predict(X_test_scaled_fm)
        
        # Métricas
        metrics_pareto_fm = {
            'r2_test': r2_score(y_test_fm, y_pred_pareto_fm),
            'rmse_test': np.sqrt(mean_squared_error(y_test_fm, y_pred_pareto_fm)),
            'mae_test': mean_absolute_error(y_test_fm, y_pred_pareto_fm)
        }
        
        results['factor_merma'] = {
            'model': lr_pareto_fm,
            'scaler': scaler_pareto_fm,
            'top_variables': top_vars_fm,
            'metrics': metrics_pareto_fm,
            'y_test': y_test_fm,
            'y_pred': y_pred_pareto_fm,
            'X_test': X_test_fm
        }
    
    return results

# --- 5. Funciones de visualización ---
def save_plot(fig, filename, dpi=200):
    """Guarda una figura con DPI específico"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def export_data_with_predictions(df_clean, vel_col, fm_col, lr_vel=None, lr_fm=None, scaler_vel=None, scaler_fm=None, X=None, pareto_results=None):
    """Exporta los datos con columnas de predicciones agregadas"""
    # Crear una copia del dataframe original
    df_export = df_clean.copy()
    
    # Eliminar la columna Fecha_entrada_MP si existe
    if 'Fecha_entrada_MP' in df_export.columns:
        df_export = df_export.drop('Fecha_entrada_MP', axis=1)
    
    # Agregar predicciones de velocidad si el modelo existe
    if lr_vel is not None and X is not None and scaler_vel is not None:
        X_scaled = scaler_vel.transform(X)
        vel_predictions = lr_vel.predict(X_scaled)
        df_export['Regresión PT'] = vel_predictions
    else:
        df_export['Regresión PT'] = None
    
    # Agregar predicciones de factor de merma si el modelo existe
    if lr_fm is not None and X is not None and scaler_fm is not None:
        X_scaled = scaler_fm.transform(X)
        fm_predictions = lr_fm.predict(X_scaled)
        df_export['Regresión FM'] = fm_predictions
    else:
        df_export['Regresión FM'] = None
    
    # Agregar predicciones de modelos Pareto si existen
    if pareto_results is not None:
        # Predicciones Pareto para velocidad
        if 'velocidad' in pareto_results:
            pareto_vel = pareto_results['velocidad']
            X_pareto_vel = X[pareto_vel['top_variables']]
            X_pareto_vel_scaled = pareto_vel['scaler'].transform(X_pareto_vel)
            vel_pareto_predictions = pareto_vel['model'].predict(X_pareto_vel_scaled)
            df_export['Regresión PT Pareto'] = vel_pareto_predictions
        else:
            df_export['Regresión PT Pareto'] = None
        
        # Predicciones Pareto para factor de merma
        if 'factor_merma' in pareto_results:
            pareto_fm = pareto_results['factor_merma']
            X_pareto_fm = X[pareto_fm['top_variables']]
            X_pareto_fm_scaled = pareto_fm['scaler'].transform(X_pareto_fm)
            fm_pareto_predictions = pareto_fm['model'].predict(X_pareto_fm_scaled)
            df_export['Regresión FM Pareto'] = fm_pareto_predictions
        else:
            df_export['Regresión FM Pareto'] = None
    
    # Exportar a Excel
    filename = 'datos_con_predicciones.xlsx'
    df_export.to_excel(filename, index=False)
    
    return filename, df_export

def create_distribution_plot(data, title, filename):
    """Crea histograma de distribución"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data.dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Valor', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.grid(True, alpha=0.3)
    save_plot(fig, filename)
    return fig

def create_correlation_matrix(X, title, filename):
    """Crea matriz de correlación"""
    # Seleccionar las 15 variables más correlacionadas
    corr_matrix = X.corr()
    # Obtener correlaciones más altas (excluyendo diagonal)
    corr_values = corr_matrix.abs().values
    np.fill_diagonal(corr_values, 0)
    top_vars = corr_matrix.columns[np.argsort(corr_values.max(axis=1))[-15:]]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix.loc[top_vars, top_vars], annot=True, cmap='coolwarm', 
                center=0, fmt='.2f', ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    save_plot(fig, filename)
    return fig

def create_pred_vs_real_plot(y_true, y_pred, r2, title, filename):
    """Crea gráfico de predicciones vs reales"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true, y_pred, alpha=0.6, color='blue')
    
    # Línea de referencia perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
    
    ax.set_xlabel('Valores Reales', fontsize=12)
    ax.set_ylabel('Valores Predichos', fontsize=12)
    ax.set_title(f'{title} (R² = {r2:.3f})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, filename)
    return fig

def create_residuals_plot(y_true, residuals, title, filename):
    """Crea gráfico de residuos"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, residuals, alpha=0.6, color='red')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.8)
    ax.set_xlabel('Valores Reales', fontsize=12)
    ax.set_ylabel('Residuos', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    save_plot(fig, filename)
    return fig

def create_importance_plot(lr_importance, title, filename):
    """Crea gráfico de importancia de variables para regresión lineal"""
    # Tomar top 15 variables
    top_lr = lr_importance.head(15)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Regresión Lineal
    ax.barh(range(len(top_lr)), top_lr.values, color='skyblue', alpha=0.8)
    ax.set_yticks(range(len(top_lr)))
    ax.set_yticklabels(top_lr.index, fontsize=10)
    ax.set_xlabel('Importancia (|Coeficiente|)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, filename)
    return fig

def create_pareto_plot(importance_data, title, filename):
    """Crea análisis de Pareto de importancia de variables"""
    # Ordenar por importancia descendente
    sorted_data = importance_data.sort_values(ascending=False)
    
    # Calcular porcentaje acumulado
    cumulative_pct = (sorted_data.cumsum() / sorted_data.sum()) * 100
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Gráfico de barras
    bars = ax1.bar(range(len(sorted_data)), sorted_data.values, 
                   color='skyblue', alpha=0.8, edgecolor='navy')
    ax1.set_xlabel('Variables MP', fontsize=12)
    ax1.set_ylabel('Importancia (Regresión Lineal)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Línea de porcentaje acumulado
    ax2 = ax1.twinx()
    ax2.plot(range(len(sorted_data)), cumulative_pct.values, 
             color='red', marker='o', linewidth=2, markersize=4)
    ax2.set_ylabel('Porcentaje Acumulado (%)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 105)
    
    # Línea del 80%
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax2.text(len(sorted_data)*0.7, 82, '80% Regla de Pareto', 
             fontsize=10, color='red', fontweight='bold')
    
    # Etiquetas del eje x
    ax1.set_xticks(range(len(sorted_data)))
    ax1.set_xticklabels(sorted_data.index, rotation=45, ha='right', fontsize=9)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_plot(fig, filename)
    return fig

# --- 6. Dashboard principal ---
def main():
    st.title("📊 Dashboard Interactivo - Análisis de Regresión Múltiple")
    st.markdown("---")
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Cargar datos
    data_file = st.sidebar.file_uploader(
        "Subir archivo Excel", 
        type=['xlsx', 'xls'],
        help="Sube cualquier archivo Excel con datos para análisis de regresión"
    )
    
    # Mostrar información sobre el formato esperado
    with st.sidebar.expander("ℹ️ Formato de datos esperado"):
        st.write("""
        **El archivo debe contener:**
        - Variables numéricas que contengan '_MP' en el nombre (variables predictoras)
        - Variables objetivo que contengan 'vel' o 'fm'/'merma' en el nombre
        - Columnas 'MateriaPrima' y 'Proveedor' (opcional)
        """)
    
    # Opción para usar archivos locales disponibles
    import glob
    local_excel_files = glob.glob("*.xlsx") + glob.glob("*.xls")
    
    if local_excel_files and data_file is None:
        st.sidebar.markdown("---")
        selected_file = st.sidebar.selectbox(
            "O selecciona un archivo local:",
            ['Ninguno'] + local_excel_files
        )
        if selected_file != 'Ninguno':
            data_file = selected_file
    
    if data_file is not None:
        # Cargar datos (función cacheada)
        with st.spinner("Cargando datos..."):
            df, error = load_data(data_file)
            
        if df is not None:
            # Mostrar interfaz de usuario para selecciones
            materia_seleccionada, proveedores_excluir, selected_targets = show_data_interface(df)
            
            if selected_targets:
                # Procesar y limpiar datos según selecciones
                with st.spinner("Procesando datos..."):
                    df_clean, mp_cols, vel_col, fm_col = process_and_clean_data(
                        df, materia_seleccionada, proveedores_excluir, selected_targets
                    )
                
                # Mostrar información de procesamiento
                if materia_seleccionada != 'Todas':
                    st.write(f"- Registros de {materia_seleccionada}: {len(df_clean)}")
                if proveedores_excluir:
                    st.write(f"- Registros después de excluir proveedores: {len(df_clean)}")
                if selected_targets:
                    st.write(f"- Registros después de eliminar nulos en objetivos: {len(df_clean)}")
                    if vel_col:
                        st.write(f"- Variable objetivo 1: {vel_col}")
                    if fm_col:
                        st.write(f"- Variable objetivo 2: {fm_col}")
                
                st.write(f"- Variables MP encontradas: {len(mp_cols)}")
                if mp_cols:
                    st.write("Variables MP:", mp_cols[:10])  # Mostrar solo las primeras 10
                    if len(mp_cols) > 10:
                        st.write(f"... y {len(mp_cols) - 10} más")
            else:
                st.error("❌ Por favor selecciona al menos una variable objetivo")
                return
        else:
            st.error(f"❌ Error al cargar el archivo: {error}")
            return
        
        if df_clean is not None and len(selected_targets) > 0:
            
            if not mp_cols:
                st.error("❌ No se encontraron variables MP válidas")
                return
            
            # Obtener características y objetivos
            X, y_vel, y_fm = get_features_targets(df_clean, mp_cols, vel_col, fm_col)
            
            # Métricas principales
            st.header("📊 Resumen de Datos")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Registros Analizados", len(df_clean))
            with col2:
                st.metric("Variables MP", len(mp_cols))
            with col3:
                st.metric("Variable Objetivo 1", vel_col if vel_col else "No seleccionada")
            with col4:
                st.metric("Variable Objetivo 2", fm_col if fm_col else "No seleccionada")
            
            # Tabs para organizar el contenido
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Modelado", "📊 Visualizaciones", "🔍 Importancia", "📋 Conclusiones"])
            
            with tab1:
                st.header("🤖 Modelado y Métricas")
                
                # Modelado para Variable Objetivo 1
                if y_vel is not None:
                    st.subheader(f"Modelo para {vel_col}")
                    with st.spinner(f"Entrenando modelo de regresión lineal para {vel_col}..."):
                        results_vel = train_models(X, y_vel, vel_col)
                    
                    if results_vel[0] is not None:
                        (lr_vel, scaler_vel, X_train_vel, X_test_vel, y_train_vel, y_test_vel,
                         y_pred_lr_vel, res_lr_vel, metrics_lr_vel) = results_vel
                        
                        # Mostrar métricas
                        st.write("**📈 Métricas del Modelo de Regresión Lineal**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("R² Test", f"{metrics_lr_vel['r2_test']:.3f}")
                        with col2:
                            st.metric("RMSE Test", f"{metrics_lr_vel['rmse_test']:.3f}")
                        with col3:
                            st.metric("MAE Test", f"{metrics_lr_vel['mae_test']:.3f}")
                
                # Modelado para Variable Objetivo 2
                if y_fm is not None:
                    st.subheader(f"Modelo para {fm_col}")
                    with st.spinner(f"Entrenando modelo de regresión lineal para {fm_col}..."):
                        results_fm = train_models(X, y_fm, fm_col)
                    
                    if results_fm[0] is not None:
                        (lr_fm, scaler_fm, X_train_fm, X_test_fm, y_train_fm, y_test_fm,
                         y_pred_lr_fm, res_lr_fm, metrics_lr_fm) = results_fm
                        
                        # Mostrar métricas
                        st.write("**📈 Métricas del Modelo de Regresión Lineal**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("R² Test", f"{metrics_lr_fm['r2_test']:.3f}")
                        with col2:
                            st.metric("RMSE Test", f"{metrics_lr_fm['rmse_test']:.3f}")
                        with col3:
                            st.metric("MAE Test", f"{metrics_lr_fm['mae_test']:.3f}")
                
                # Modelos Pareto (6 variables más importantes)
                st.header("🎯 Modelos con Variables Pareto (Top 6)")
                st.markdown("*Modelos entrenados usando solo las 6 variables más importantes del análisis de Pareto*")
                
                # Entrenar modelos Pareto si tenemos los modelos principales
                pareto_results = None
                if ('results_vel' in locals() and results_vel[0] is not None) or ('results_fm' in locals() and results_fm[0] is not None):
                    with st.spinner("Entrenando modelos con variables Pareto..."):
                        lr_vel_model = lr_vel if 'lr_vel' in locals() else None
                        lr_fm_model = lr_fm if 'lr_fm' in locals() else None
                        scaler_vel_model = scaler_vel if 'scaler_vel' in locals() else None
                        scaler_fm_model = scaler_fm if 'scaler_fm' in locals() else None
                        
                        pareto_results = train_pareto_models(
                            X, y_vel, y_fm, mp_cols, 
                            scaler_vel_model, scaler_fm_model, 
                            lr_vel_model, lr_fm_model
                        )
                
                # Mostrar resultados de modelos Pareto
                if pareto_results:
                    col1, col2 = st.columns(2)
                    
                    # Velocidad Pareto
                    if 'velocidad' in pareto_results:
                        with col1:
                            st.subheader(f"🚀 {vel_col} - Modelo Pareto")
                            pareto_vel = pareto_results['velocidad']
                            
                            # Variables seleccionadas
                            st.write("**Variables más importantes:**")
                            for i, var in enumerate(pareto_vel['top_variables'], 1):
                                st.write(f"{i}. {var}")
                            
                            # Métricas Pareto
                            st.write("**📈 Métricas del Modelo Pareto**")
                            subcol1, subcol2, subcol3 = st.columns(3)
                            with subcol1:
                                st.metric("R² Test", f"{pareto_vel['metrics']['r2_test']:.3f}")
                            with subcol2:
                                st.metric("RMSE Test", f"{pareto_vel['metrics']['rmse_test']:.3f}")
                            with subcol3:
                                st.metric("MAE Test", f"{pareto_vel['metrics']['mae_test']:.3f}")
                            
                            # Comparación con modelo completo
                            if 'metrics_lr_vel' in locals():
                                st.write("**📊 Comparación con Modelo Completo**")
                                delta_r2 = pareto_vel['metrics']['r2_test'] - metrics_lr_vel['r2_test']
                                delta_rmse = pareto_vel['metrics']['rmse_test'] - metrics_lr_vel['rmse_test']
                                st.metric("Δ R²", f"{delta_r2:+.3f}", delta=f"{delta_r2:+.3f}")
                                st.metric("Δ RMSE", f"{delta_rmse:+.3f}", delta=f"{delta_rmse:+.3f}")
                    
                    # Factor de Merma Pareto
                    if 'factor_merma' in pareto_results:
                        with col2:
                            st.subheader(f"📉 {fm_col} - Modelo Pareto")
                            pareto_fm = pareto_results['factor_merma']
                            
                            # Variables seleccionadas
                            st.write("**Variables más importantes:**")
                            for i, var in enumerate(pareto_fm['top_variables'], 1):
                                st.write(f"{i}. {var}")
                            
                            # Métricas Pareto
                            st.write("**📈 Métricas del Modelo Pareto**")
                            subcol1, subcol2, subcol3 = st.columns(3)
                            with subcol1:
                                st.metric("R² Test", f"{pareto_fm['metrics']['r2_test']:.3f}")
                            with subcol2:
                                st.metric("RMSE Test", f"{pareto_fm['metrics']['rmse_test']:.3f}")
                            with subcol3:
                                st.metric("MAE Test", f"{pareto_fm['metrics']['mae_test']:.3f}")
                            
                            # Comparación con modelo completo
                            if 'metrics_lr_fm' in locals():
                                st.write("**📊 Comparación con Modelo Completo**")
                                delta_r2 = pareto_fm['metrics']['r2_test'] - metrics_lr_fm['r2_test']
                                delta_rmse = pareto_fm['metrics']['rmse_test'] - metrics_lr_fm['rmse_test']
                                st.metric("Δ R²", f"{delta_r2:+.3f}", delta=f"{delta_r2:+.3f}")
                                st.metric("Δ RMSE", f"{delta_rmse:+.3f}", delta=f"{delta_rmse:+.3f}")
            
            with tab2:
                st.header("📊 Visualizaciones")
                
                # Generar todas las gráficas
                with st.spinner("Generando visualizaciones..."):
                    
                    # G1 y G2: Distribuciones
                    if y_vel is not None:
                        st.subheader("Distribución de Velocidad")
                        fig1 = create_distribution_plot(y_vel, "Distribución de Velocidad de Procesamiento", "dash_dist_velpt.png")
                        st.pyplot(fig1)
                    
                    if y_fm is not None:
                        st.subheader("Distribución de Factor de Merma")
                        fig2 = create_distribution_plot(y_fm, "Distribución de Factor de Merma", "dash_dist_fm.png")
                        st.pyplot(fig2)
                    
                    # G3: Matriz de correlación
                    st.subheader("Matriz de Correlación")
                    fig3 = create_correlation_matrix(X, "Matriz de Correlación - Variables MP", "dash_corr_matrix.png")
                    st.pyplot(fig3)
                    
                    # G4 y G5: Predicciones vs Reales
                    if 'results_vel' in locals() and results_vel[0] is not None:
                        st.subheader("Predicciones vs Reales - Velocidad")
                        fig4 = create_pred_vs_real_plot(y_test_vel, y_pred_lr_vel, metrics_lr_vel['r2_test'],
                                                       "Predicciones vs Reales - Velocidad (Regresión Lineal)", "dash_pred_vs_real_velpt.png")
                        st.pyplot(fig4)
                    
                    if 'results_fm' in locals() and results_fm[0] is not None:
                        st.subheader("Predicciones vs Reales - Factor de Merma")
                        fig5 = create_pred_vs_real_plot(y_test_fm, y_pred_lr_fm, metrics_lr_fm['r2_test'],
                                                       "Predicciones vs Reales - Factor de Merma (Regresión Lineal)", "dash_pred_vs_real_fm.png")
                        st.pyplot(fig5)
                    
                    # G6: Residuos
                    st.subheader("Análisis de Residuos")
                    if 'results_vel' in locals() and results_vel[0] is not None:
                        fig6a = create_residuals_plot(y_test_vel, res_lr_vel, "Residuos - Modelo Velocidad", "dash_residuos_vel.png")
                        st.pyplot(fig6a)
                    
                    if 'results_fm' in locals() and results_fm[0] is not None:
                        fig6b = create_residuals_plot(y_test_fm, res_lr_fm, "Residuos - Modelo Factor de Merma", "dash_residuos_fm.png")
                        st.pyplot(fig6b)
                    
                    # Visualizaciones para modelos Pareto
                    if 'pareto_results' in locals() and pareto_results:
                        st.header("🎯 Visualizaciones - Modelos Pareto")
                        
                        # Predicciones vs Reales para modelos Pareto
                        if 'velocidad' in pareto_results:
                            st.subheader("Predicciones vs Reales - Velocidad (Modelo Pareto)")
                            pareto_vel = pareto_results['velocidad']
                            fig_pareto_vel = create_pred_vs_real_plot(
                                pareto_vel['y_test'], pareto_vel['y_pred'], 
                                pareto_vel['metrics']['r2_test'],
                                "Predicciones vs Reales - Velocidad (Modelo Pareto - Top 6 Variables)", 
                                "dash_pred_vs_real_velpt_pareto.png"
                            )
                            st.pyplot(fig_pareto_vel)
                        
                        if 'factor_merma' in pareto_results:
                            st.subheader("Predicciones vs Reales - Factor de Merma (Modelo Pareto)")
                            pareto_fm = pareto_results['factor_merma']
                            fig_pareto_fm = create_pred_vs_real_plot(
                                pareto_fm['y_test'], pareto_fm['y_pred'], 
                                pareto_fm['metrics']['r2_test'],
                                "Predicciones vs Reales - Factor de Merma (Modelo Pareto - Top 6 Variables)", 
                                "dash_pred_vs_real_fm_pareto.png"
                            )
                            st.pyplot(fig_pareto_fm)
                        
                        # Residuos para modelos Pareto
                        st.subheader("Análisis de Residuos - Modelos Pareto")
                        if 'velocidad' in pareto_results:
                            pareto_vel = pareto_results['velocidad']
                            residuos_pareto_vel = pareto_vel['y_test'] - pareto_vel['y_pred']
                            fig_res_pareto_vel = create_residuals_plot(
                                pareto_vel['y_test'], residuos_pareto_vel, 
                                "Residuos - Modelo Velocidad Pareto (Top 6 Variables)", 
                                "dash_residuos_vel_pareto.png"
                            )
                            st.pyplot(fig_res_pareto_vel)
                        
                        if 'factor_merma' in pareto_results:
                            pareto_fm = pareto_results['factor_merma']
                            residuos_pareto_fm = pareto_fm['y_test'] - pareto_fm['y_pred']
                            fig_res_pareto_fm = create_residuals_plot(
                                pareto_fm['y_test'], residuos_pareto_fm, 
                                "Residuos - Modelo Factor de Merma Pareto (Top 6 Variables)", 
                                "dash_residuos_fm_pareto.png"
                            )
                            st.pyplot(fig_res_pareto_fm)
            
            with tab3:
                st.header("🔍 Análisis de Importancia")
                
                # Análisis de importancia
                if 'results_vel' in locals() and results_vel[0] is not None:
                    st.subheader("Importancia de Variables - Velocidad")
                    lr_imp_vel = get_feature_importance(lr_vel, mp_cols, scaler_vel)
                    
                    fig7a = create_importance_plot(lr_imp_vel, "Importancia de Variables - Velocidad", "dash_importancia_vel.png")
                    st.pyplot(fig7a)
                    
                    # Pareto para velocidad
                    fig8a = create_pareto_plot(lr_imp_vel, "Análisis de Pareto - Variables MP (Velocidad)", "dash_pareto_vel.png")
                    st.pyplot(fig8a)
                
                if 'results_fm' in locals() and results_fm[0] is not None:
                    st.subheader("Importancia de Variables - Factor de Merma")
                    lr_imp_fm = get_feature_importance(lr_fm, mp_cols, scaler_fm)
                    
                    fig7b = create_importance_plot(lr_imp_fm, "Importancia de Variables - Factor de Merma", "dash_importancia_fm.png")
                    st.pyplot(fig7b)
                    
                    # Pareto para factor de merma
                    fig8b = create_pareto_plot(lr_imp_fm, "Análisis de Pareto - Variables MP (Factor de Merma)", "dash_pareto_fm.png")
                    st.pyplot(fig8b)
                
                # Crear Pareto combinado
                if 'lr_imp_vel' in locals() and 'lr_imp_fm' in locals():
                    # Combinar importancias (promedio)
                    combined_importance = (lr_imp_vel + lr_imp_fm) / 2
                    combined_importance = combined_importance.sort_values(ascending=False)
                    fig_pareto = create_pareto_plot(combined_importance, "Análisis de Pareto - Variables MP Más Influyentes", "dash_pareto.png")
                    st.pyplot(fig_pareto)
            
            with tab4:
                st.header("📋 Conclusiones Automáticas")
                
                # Generar conclusiones
                conclusions = []
                
                # Resultados de los modelos
                if 'metrics_lr_vel' in locals():
                    conclusions.append(f"**Velocidad**: Modelo de Regresión Lineal con R² = {metrics_lr_vel['r2_test']:.3f}")
                
                if 'metrics_lr_fm' in locals():
                    conclusions.append(f"**Factor de Merma**: Modelo de Regresión Lineal con R² = {metrics_lr_fm['r2_test']:.3f}")
                
                # Variables más influyentes
                if 'combined_importance' in locals():
                    top_5_vars = combined_importance.head(5)
                    conclusions.append("**Top 5 Variables MP más influyentes:**")
                    for i, (var, importance) in enumerate(top_5_vars.items(), 1):
                        conclusions.append(f"{i}. {var}: {importance:.3f}")
                
                # Correlaciones más fuertes
                corr_with_targets = []
                if y_vel is not None:
                    vel_corr = X.corrwith(y_vel).abs().sort_values(ascending=False).head(3)
                    corr_with_targets.extend([f"Velocidad-{var}: {corr:.3f}" for var, corr in vel_corr.items()])
                
                if y_fm is not None:
                    fm_corr = X.corrwith(y_fm).abs().sort_values(ascending=False).head(3)
                    corr_with_targets.extend([f"Factor Merma-{var}: {corr:.3f}" for var, corr in fm_corr.items()])
                
                if corr_with_targets:
                    conclusions.append("**Correlaciones más fuertes:**")
                    conclusions.extend(corr_with_targets[:5])
                
                # Mostrar conclusiones
                for conclusion in conclusions:
                    st.write(conclusion)
                
                # Mostrar ecuaciones de regresión
                st.subheader("📊 Ecuaciones de Regresión")
                
                if 'lr_vel' in locals() and lr_vel is not None:
                    # Obtener coeficientes de regresión para Velocidad PT
                    coef_vel = lr_vel.coef_
                    intercept_vel = lr_vel.intercept_
                    
                    # Crear ecuación de regresión para Velocidad PT
                    equation_vel = f"Velocidad PT = {intercept_vel:.4f}"
                    for i, coef in enumerate(coef_vel):
                        if coef >= 0:
                            equation_vel += f" + {coef:.4f} * {X.columns[i]}"
                        else:
                            equation_vel += f" - {abs(coef):.4f} * {X.columns[i]}"
                    
                    st.write("**Ecuación de Regresión Lineal para Velocidad PT:**")
                    st.code(equation_vel, language="text")
                
                if 'lr_fm' in locals() and lr_fm is not None:
                    # Obtener coeficientes de regresión para Factor de Merma
                    coef_fm = lr_fm.coef_
                    intercept_fm = lr_fm.intercept_
                    
                    # Crear ecuación de regresión para Factor de Merma
                    equation_fm = f"Factor de Merma = {intercept_fm:.4f}"
                    for i, coef in enumerate(coef_fm):
                        if coef >= 0:
                            equation_fm += f" + {coef:.4f} * {X.columns[i]}"
                        else:
                            equation_fm += f" - {abs(coef):.4f} * {X.columns[i]}"
                    
                    st.write("**Ecuación de Regresión Lineal para Factor de Merma:**")
                    st.code(equation_fm, language="text")
                
                # Ecuaciones de regresión para modelos Pareto
                if 'pareto_results' in locals() and pareto_results:
                    st.subheader("🎯 Ecuaciones de Regresión - Modelos Pareto (Top 6 Variables)")
                    
                    if 'velocidad' in pareto_results:
                        pareto_vel = pareto_results['velocidad']
                        lr_pareto_vel = pareto_vel['model']
                        top_vars_vel = pareto_vel['top_variables']
                        
                        # Crear ecuación de regresión Pareto para Velocidad
                        coef_pareto_vel = lr_pareto_vel.coef_
                        intercept_pareto_vel = lr_pareto_vel.intercept_
                        
                        equation_pareto_vel = f"Velocidad PT (Pareto) = {intercept_pareto_vel:.4f}"
                        for i, coef in enumerate(coef_pareto_vel):
                            if coef >= 0:
                                equation_pareto_vel += f" + {coef:.4f} * {top_vars_vel[i]}"
                            else:
                                equation_pareto_vel += f" - {abs(coef):.4f} * {top_vars_vel[i]}"
                        
                        st.write("**Ecuación de Regresión Pareto para Velocidad PT:**")
                        st.code(equation_pareto_vel, language="text")
                    
                    if 'factor_merma' in pareto_results:
                        pareto_fm = pareto_results['factor_merma']
                        lr_pareto_fm = pareto_fm['model']
                        top_vars_fm = pareto_fm['top_variables']
                        
                        # Crear ecuación de regresión Pareto para Factor de Merma
                        coef_pareto_fm = lr_pareto_fm.coef_
                        intercept_pareto_fm = lr_pareto_fm.intercept_
                        
                        equation_pareto_fm = f"Factor de Merma (Pareto) = {intercept_pareto_fm:.4f}"
                        for i, coef in enumerate(coef_pareto_fm):
                            if coef >= 0:
                                equation_pareto_fm += f" + {coef:.4f} * {top_vars_fm[i]}"
                            else:
                                equation_pareto_fm += f" - {abs(coef):.4f} * {top_vars_fm[i]}"
                        
                        st.write("**Ecuación de Regresión Pareto para Factor de Merma:**")
                        st.code(equation_pareto_fm, language="text")
                    
                    # Resumen comparativo
                    st.subheader("📈 Resumen Comparativo: Modelo Completo vs Modelo Pareto")
                    
                    if 'velocidad' in pareto_results and 'metrics_lr_vel' in locals():
                        st.write("**Velocidad PT:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("*Modelo Completo:*")
                            st.write(f"- R²: {metrics_lr_vel['r2_test']:.3f}")
                            st.write(f"- RMSE: {metrics_lr_vel['rmse_test']:.3f}")
                            st.write(f"- Variables: {len(mp_cols)}")
                        with col2:
                            st.write("*Modelo Pareto:*")
                            pareto_vel_metrics = pareto_results['velocidad']['metrics']
                            st.write(f"- R²: {pareto_vel_metrics['r2_test']:.3f}")
                            st.write(f"- RMSE: {pareto_vel_metrics['rmse_test']:.3f}")
                            st.write(f"- Variables: 6")
                    
                    if 'factor_merma' in pareto_results and 'metrics_lr_fm' in locals():
                        st.write("**Factor de Merma:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("*Modelo Completo:*")
                            st.write(f"- R²: {metrics_lr_fm['r2_test']:.3f}")
                            st.write(f"- RMSE: {metrics_lr_fm['rmse_test']:.3f}")
                            st.write(f"- Variables: {len(mp_cols)}")
                        with col2:
                            st.write("*Modelo Pareto:*")
                            pareto_fm_metrics = pareto_results['factor_merma']['metrics']
                            st.write(f"- R²: {pareto_fm_metrics['r2_test']:.3f}")
                            st.write(f"- RMSE: {pareto_fm_metrics['rmse_test']:.3f}")
                            st.write(f"- Variables: 6")
                
                # Exportar datos con predicciones
                st.subheader("📁 Exportar Datos")
                
                # Preparar modelos para exportación
                lr_vel_export = lr_vel if 'lr_vel' in locals() else None
                lr_fm_export = lr_fm if 'lr_fm' in locals() else None
                scaler_vel_export = scaler_vel if 'scaler_vel' in locals() else None
                scaler_fm_export = scaler_fm if 'scaler_fm' in locals() else None
                pareto_results_export = pareto_results if 'pareto_results' in locals() else None
                
                try:
                    filename, df_exported = export_data_with_predictions(
                        df_clean, vel_col, fm_col, 
                        lr_vel_export, lr_fm_export,
                        scaler_vel_export, scaler_fm_export, X,
                        pareto_results_export
                    )
                    
                    st.success(f"✅ Datos exportados exitosamente: {filename}")
                    st.write(f"📊 Registros exportados: {len(df_exported)}")
                    st.write(f"📋 Columnas incluidas: {len(df_exported.columns)}")
                    
                    # Botón de descarga del archivo Excel
                    with open(filename, "rb") as file:
                        btn = st.download_button(
                            label="📥 Descargar archivo Excel con predicciones",
                            data=file,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        if btn:
                            st.success("¡Archivo descargado exitosamente!")
                    
                    # Mostrar preview de las nuevas columnas
                    if 'Regresión PT' in df_exported.columns or 'Regresión FM' in df_exported.columns:
                        st.write("**Nuevas columnas agregadas:**")
                        preview_cols = []
                        if 'Regresión PT' in df_exported.columns:
                            preview_cols.append('Regresión PT')
                        if 'Regresión FM' in df_exported.columns:
                            preview_cols.append('Regresión FM')
                        
                        if vel_col:
                            preview_cols.insert(0, vel_col)
                        if fm_col:
                            preview_cols.append(fm_col)
                        
                        st.dataframe(df_exported[preview_cols].head(10))
                    
                except Exception as e:
                    st.error(f"❌ Error al exportar datos: {str(e)}")
                
                # Resumen final en consola
                st.subheader("📄 Resumen para Consola")
                console_summary = f"""
                ============================================================
                    RESUMEN FINAL - ANÁLISIS VELOCIDADES ESPINACA
                ============================================================
                
                Registros analizados: {len(df_clean)}
                Variables MP utilizadas: {len(mp_cols)}
                
                MÉTRICAS DE MODELOS:
                """
                
                if 'metrics_lr_vel' in locals():
                    console_summary += f"""
                Velocidad:
                - Regresión Lineal R²: {metrics_lr_vel['r2_test']:.3f}
                - RMSE: {metrics_lr_vel['rmse_test']:.3f}
                - MAE: {metrics_lr_vel['mae_test']:.3f}
                """
                
                if 'metrics_lr_fm' in locals():
                    console_summary += f"""
                Factor de Merma:
                - Regresión Lineal R²: {metrics_lr_fm['r2_test']:.3f}
                - RMSE: {metrics_lr_fm['rmse_test']:.3f}
                - MAE: {metrics_lr_fm['mae_test']:.3f}
                """
                
                if 'top_5_vars' in locals():
                    console_summary += "\nTOP 5 VARIABLES MP MÁS INFLUYENTES:\n"
                    for i, (var, importance) in enumerate(top_5_vars.items(), 1):
                        console_summary += f"{i}. {var}: {importance:.3f}\n"
                
                console_summary += """
                
                ARCHIVOS GENERADOS:
                - dash_dist_velpt.png
                - dash_dist_fm.png
                - dash_corr_matrix.png
                - dash_pred_vs_real_velpt.png
                - dash_pred_vs_real_fm.png
                - dash_importancia.png
                - dash_pareto.png
                - datos_con_predicciones.xlsx (con columnas de predicciones de regresión)
                
                ============================================================
                """
                
                st.code(console_summary)
                
                # Imprimir en consola también
                print(console_summary)
    
    else:
        st.info("👆 Por favor, sube un archivo Excel o selecciona uno de los archivos locales disponibles.")
        st.markdown("""
        ### 📋 Instrucciones:
        1. Sube cualquier archivo de Excel usando el botón de arriba
        2. El sistema automáticamente:
           - Detectará las variables predictoras (columnas con '_MP')
           - Identificará las variables objetivo (columnas con 'vel', 'fm', 'merma')
           - Permitirá filtrar por materia prima y proveedores
           - Limpiará los datos y manejará valores nulos
           - Entrenará modelos de regresión lineal
           - Generará visualizaciones interactivas
           - Mostrará análisis de importancia y conclusiones
        
        ### 🎯 Funcionalidades:
        - **Modelado**: Regresión Lineal para variables objetivo detectadas automáticamente
        - **Visualizaciones**: Gráficas interactivas guardadas automáticamente
        - **Análisis**: Importancia de variables y análisis de Pareto
        - **Conclusiones**: Resumen automático con métricas y recomendaciones
        - **Flexibilidad**: Compatible con cualquier archivo Excel que siga la estructura esperada
        """)

if __name__ == "__main__":
    main()