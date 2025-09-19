import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import plotly.express as px

# Función para convertir distancia en metros a acres
def distancia_a_acres(distancia):
    radio_metros = distancia / 2
    area_metros_cuadrados = np.pi * (radio_metros**2)
    area_acres = area_metros_cuadrados / 4046.86
    return area_acres

# Función para validar que los rangos sean crecientes
def validar_rangos(rangos):
    for i in range(len(rangos) - 1):
        if rangos[i][1] > rangos[i + 1][0]:
            return False
    return True

# Streamlit UI
st.image("OIG.jpg", width=200)
st.title("Well Clustering for Workover Candidate Selection")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("dfx.csv", sep=";")

# Parámetros para la clasificación de espaciamiento
st.sidebar.header("Parámetros de Clasificación por Espaciamiento (Acres)")
num_rangos_esp = st.sidebar.number_input("Número de Rangos (Espaciamiento, incluyendo extremos)", min_value=1, max_value=10, value=5, step=1)

rangos_esp = []
for i in range(num_rangos_esp):
    col1, col2 = st.sidebar.columns(2)
    if i == 0:
        fin = col2.number_input(f"Fin del Rango Menor que (Acres)", min_value=0.0, value=4.0, step=0.1, key=f"esp_fin_{i}")
        rangos_esp.append((-np.inf, fin))
    elif i == num_rangos_esp - 1:
        inicio = col1.number_input(f"Inicio del Rango Mayor que (Acres)", min_value=rangos_esp[-1][1], value=rangos_esp[-1][1] + 0.1, step=0.1, key=f"esp_inicio_{i}")
        rangos_esp.append((inicio, np.inf))
    else:
        inicio = col1.number_input(f"Inicio Rango {i} (Acres)", min_value=rangos_esp[-1][1], value=rangos_esp[-1][1] + 0.1, step=0.1, key=f"esp_inicio_{i}")
        fin = col2.number_input(f"Fin Rango {i} (Acres)", min_value=inicio, value=inicio + 2.0, step=0.1, key=f"esp_fin_{i}")
        rangos_esp.append((inicio, fin))

if not validar_rangos(rangos_esp):
    st.sidebar.error("Los rangos deben ser crecientes y no superponerse. Por favor, corrige los valores.")

# Parámetros para la clasificación por volumen acumulado
st.sidebar.header("Parámetros de Clasificación por Volumen Acumulado (Np)")
num_rangos_vol = st.sidebar.number_input("Número de Rangos (Volumen, incluyendo extremos)", min_value=1, max_value=10, value=3, step=1)

rangos_vol = []
for i in range(num_rangos_vol):
    col1, col2 = st.sidebar.columns(2)
    if i == 0:
        fin = col2.number_input(f"Fin del Rango Menor que (Np)", min_value=0.0, value=5.0, step=0.1, key=f"vol_fin_{i}")
        rangos_vol.append((-np.inf, fin))
    elif i == num_rangos_vol - 1:
        inicio = col1.number_input(f"Inicio del Rango Mayor que (Np)", min_value=rangos_vol[-1][1], value=rangos_vol[-1][1] + 0.1, step=0.1, key=f"vol_inicio_{i}")
        rangos_vol.append((inicio, np.inf))
    else:
        inicio = col1.number_input(f"Inicio Rango {i} (Np)", min_value=rangos_vol[-1][1], value=rangos_vol[-1][1] + 0.1, step=0.1, key=f"vol_inicio_{i}")
        fin = col2.number_input(f"Fin Rango {i} (Np)", min_value=inicio, value=inicio + 5.0, step=0.1, key=f"vol_fin_{i}")
        rangos_vol.append((inicio, fin))

if not validar_rangos(rangos_vol):
    st.sidebar.error("Los rangos deben ser crecientes y no superponerse. Por favor, corrige los valores.")

# Clasificación de pozos por zona
st.sidebar.header("Clustering por Zona")
zonas = df['Zone Name'].unique()
dataframes_zonas = {}

for zona in zonas:
    df_zona = df[df['Zone Name'] == zona].copy()
    
    coords = df_zona[['X', 'Y']].values
    tree = cKDTree(coords)
    distancias, _ = tree.query(coords, k=2)  # Distancia mínima a otro pozo
    df_zona['distancia_min'] = distancias[:, 1]
    df_zona['espaciamiento_acres'] = df_zona['distancia_min'].apply(distancia_a_acres)

    # Clasificación por espaciamiento
    if len(rangos_esp) > 1:
        df_zona['grupo_espaciamiento'] = pd.cut(
            df_zona['espaciamiento_acres'],
            bins=[limite[0] for limite in rangos_esp] + [np.inf],
            labels=[
                f"menor que {rangos_esp[0][1]} acres"
            ] + [
                f"entre {limite[0]} acres y {limite[1]} acres" for limite in rangos_esp[1:-1]
            ] + [
                f"mayor que {rangos_esp[-1][0]} acres"
            ],
        )
    else:
        df_zona['grupo_espaciamiento'] = f"menor que {rangos_esp[0][1]} acres"

    # Clasificación por volumen
    if len(rangos_vol) > 1:
        df_zona['grupo_volumen'] = pd.cut(
            df_zona['Cum'],
            bins=[limite[0] for limite in rangos_vol] + [np.inf],
            labels=[
                f"menor que {rangos_vol[0][1]} MSTB"
            ] + [
                f"entre {limite[0]} MSTB y {limite[1]} MSTB" for limite in rangos_vol[1:-1]
            ] + [
                f"mayor que {rangos_vol[-1][0]} MSTB"
            ],
        )
    else:
        df_zona['grupo_volumen'] = f"menor que {rangos_vol[0][1]} MSTB"

    # Clustering cruzado
    df_zona['grupo_combined'] = (
        df_zona['grupo_espaciamiento'].astype(str) + ' & ' + df_zona['grupo_volumen'].astype(str)
    )

    dataframes_zonas[zona] = df_zona

# Selección y visualización
zona_seleccionada = st.selectbox("Selecciona una Zona", zonas)
if zona_seleccionada:
    df_zona = dataframes_zonas[zona_seleccionada]
    
    fig = px.scatter(
        df_zona, x='X', y='Y', color='grupo_combined',
        title=f'Clusters para {zona_seleccionada}'
    )
    st.plotly_chart(fig)

    cluster_seleccionado = st.selectbox(
        "Selecciona un Cluster", df_zona['grupo_combined'].unique()
    )
    if cluster_seleccionado:
        df_filtrado = df_zona[df_zona['grupo_combined'] == cluster_seleccionado]
        st.write(df_filtrado)

# Exportación
if st.button("Exportar Resultados"):
    for zona, df_zona in dataframes_zonas.items():
        df_zona.to_csv(f'{zona}_clusters.csv', index=False)
    st.success("Resultados exportados exitosamente.")
