#---------------------------------------------------------------------------------------
#                                clear and specification of data
#---------------------------------------------------------------------------------------

#       load data to use
#---------------------------------------------------------------------------------------
import os
import pandas as pd
import re
import unicodedata
import unidecode
from rapidfuzz import process, fuzz

ruta_bases = r"C:\Users\User\Desktop\EconoAvanz\Trabajo_Final\bases" 
ruta_resultados = r"C:\Users\User\Desktop\EconoAvanz\Trabajo_Final\resultados" 


# Cargar las bases de datos
df_2021_2024 = pd.read_csv(os.path.join(ruta_bases, "Inclusion_Financiera_2021_2024.csv"))
df_2017_2021 = pd.read_csv(os.path.join(ruta_bases, "Inclusion_Financiera_2017_2021.csv"))
cobertura_movil = pd.read_csv(os.path.join(ruta_bases, "Cobertura_movil_por_tecnologia_y_proveedor_20250521.csv"))
internet_fijo = pd.read_csv(os.path.join(ruta_bases, "Internet_Fijo_Penetracion_Municipio_20250521.csv"))

#---------------------------------------------------------------------------------------
cols_2021_2024 = set(df_2021_2024.columns)
cols_2017_2021 = set(df_2017_2021.columns)

cols_cobertura_movil = set(cobertura_movil.columns)
cols_internet_fijo = set(internet_fijo.columns)

#---------------------------------------------------------------------------------------
#Inclusion financiera 2017_2021
df_2017_2021.columns

df_2017_2021['NRO CREDITO CONSUMO MUJERES'] = df_2017_2021['NRO TOTAL CREDITO CONSUMO'] - df_2017_2021['NRO CREDITO CONSUMO HOMBRES']
#df_2017_2021[df_2017_2021['NRO CREDITO CONSUMO MUJERES'] > 5]['NRO CREDITO CONSUMO MUJERES'] #Recuperacion de variable faltante

columnas_a_sumar_general_A = ['NRO CTA AHORRO ACTIVAS', 'NRO CTA AHORRO ELECTRONICAS ACTIVAS','NRO TOTAL CREDITO CONSUMO','NRO TOTAL CRED CONS BAJO MONTO',
                            'NRO TOTAL CREDITO VIVIENDA','NRO TOTAL MICROCREDITO ']
columnas_a_sumar_men_A = ['NRO CTA AHORRO HOMBRES','NRO CTA AHORRO ELECTRONICAS HOMBRES','NRO CREDITO CONSUMO HOMBRES','NRO CRED CONS BAJO MONTO HOMBRES',
                        'NRO CREDITO VIVIENDA HOMBRES','NRO MICROCREDITO HOMBRES']
columnas_a_sumar_women_A = ['NRO CTA AHORRO MUJERES','NRO CTA AHORRO ELECTRONICAS MUJERES','NRO CREDITO CONSUMO MUJERES','NRO CRED CONS BAJO MONTO MUJERES',
                          'NRO CREDITO VIVIENDA MUJERES','NRO MICROCREDITO MUJERES']


df_2017_2021['Inclusion_financiera_general'] = df_2017_2021[columnas_a_sumar_general_A].sum(axis=1)
df_2017_2021['Inclusion_financiera_men'] = df_2017_2021[columnas_a_sumar_men_A].sum(axis=1)
df_2017_2021['Inclusion_financiera_women'] = df_2017_2021[columnas_a_sumar_women_A].sum(axis=1)

Columns_analysis_df_2017_2021 = ['TIPO DE ENTIDAD', 'CODIGO DE LA  ENTIDAD', 'NOMBRE DE LA  ENTIDAD',
       'FECHA DE CORTE', 'UNIDAD DE CAPTURA', 'DEPARTAMENTO','MUNICIPIO', 'TIPO',
       'Inclusion_financiera_general','Inclusion_financiera_men','Inclusion_financiera_women',          
       #Nro de oficinas por cada rubro---------------------------------------------------------------
       'NRO CORRESPONSALES PROPIOS', 'NRO CORRESPONSALES TERCERIZADOS', 'NRO CORRESPONSALES ACTIVOS',
       'NRO CORRESPONSALES'] 

df_filtrado_2017_2021 = df_2017_2021[Columns_analysis_df_2017_2021]
df_filtrado_2017_2021.isna().sum()
df_filtrado_2017_2021 = df_filtrado_2017_2021.dropna()
df_filtrado_2017_2021.columns
df_filtrado_2017_2021['MUNICIPIO']
df_filtrado_2017_2021['FECHA DE CORTE']

#-----------------------------------
#Agrupamiento de municipios en común
#-----------------------------------

#---------------------------------------------------------------------------------------
#Inclusion financiera 2021_2024
df_2021_2024.columns
df_2021_2024.rename(columns={'DESC_RENGLON': 'MUNICIPIO'}, inplace=True) #Renombrar la variable que debe ser el municipio a filtrar

columnas_a_sumar_general = ['(25) NRO_CTAS_AHORRO_ACTIVAS', '(45) NRO_CRÉDITO_CONSUMO', '(51) NRO_CRED_CONS_BAJO_MONTO',
                            '(57) NRO_CRÉDITO_VIVIENDA','(75) NRO_MICROCRÉDITO',]
columnas_a_sumar_men = ['(29) NRO_CTAS_AHORRO_HOMBRES', '(43) NRO_CRÉDITO_CONSUMO_HOMBRES', '(49) NRO_CRED_CONS_BAJO_MONTO_HOMBRES',
                         '(55) NRO_CRÉDITO_VIVIENDA_HOMBRES','(73) NRO_MICROCRÉDITO_HOMBRES',]
columnas_a_sumar_women = ['(27) NRO_CTAS_AHORRO_MUJERES', '(41) NRO_CRÉDITO_CONSUMO_MUJERES', '(47) NRO_CRED_CONS_BAJO_MONTO_MUJERES',
                          '(53) NRO_CRÉDITO_VIVIENDA_MUJERES','(71) NRO_MICROCRÉDITO_MUJERES',]


df_2021_2024['Inclusion_financiera_general'] = df_2021_2024[columnas_a_sumar_general].sum(axis=1)
df_2021_2024['Inclusion_financiera_men'] = df_2021_2024[columnas_a_sumar_men].sum(axis=1)
df_2021_2024['Inclusion_financiera_women'] = df_2021_2024[columnas_a_sumar_women].sum(axis=1)

Columns_analysis_df_2021_2024 = ['TIPO_ENTIDAD', 'CODIGO_ENTIDAD', 'NOMBRE_ENTIDAD', 'FECHA_CORTE',
       'UNICAP', 'MUNICIPIO','DESCRIP_UC', 'TIPO','Inclusion_financiera_general',
       'Inclusion_financiera_men','Inclusion_financiera_women',
       #Nro de oficinas por cada rubro---------------------------------------------------------------
       '(79) NRO_CORRESPONSALES_FÍSICOS_PROPIOS_ACTIVOS',
       '(80) NRO_CORRESPONSALES_FÍSICOS_TERCERIZADOS_ACTIVOS','(81) NRO_CORRESPONSALES_PROPIOS_MÓVILES',
       '(82) NRO_CORRESPONSALES_TERCERIZADOS_MÓVILES','(83) NRO_CORRESPONSALES_MÓVILES_ACTIVOS',
       '(84) NRO_CORRESPONSALES_MÓVILES','(4) NRO_CORRESPONSALES_FÍSICOS','(1) NRO_CORRESPONSALES_FÍSICOS_PROPIOS'] 
df_filtrado_2021_2024 = df_2021_2024[Columns_analysis_df_2021_2024]
df_filtrado_2021_2024.isna().sum()
df_filtrado_2021_2024 = df_filtrado_2021_2024.dropna()

#---------------------------------------------------------------------------------------

#-----------------------------------
#Agrupamiento de municipios en común
#-----------------------------------

df_filtrado_2021_2024['MUNICIPIO'].unique()
df_filtrado_2017_2021['MUNICIPIO'].unique().shape
df_filtrado_2017_2021['MUNICIPIO'].unique().shape

df_filtrado_2017_2021['FECHA DE CORTE']
df_filtrado_2021_2024['FECHA_CORTE']

df_filtrado_2017_2021.rename(columns={'FECHA DE CORTE': 'FECHA_CORTE'}, inplace=True)
df_filtrado_2017_2021.rename(columns={'CODIGO DE LA  ENTIDAD': 'CODIGO_ENTIDAD'}, inplace=True)
df_filtrado_2017_2021.rename(columns={'NOMBRE DE LA  ENTIDAD': 'NOMBRE_ENTIDAD'}, inplace=True)
df_filtrado_2017_2021.rename(columns={'UNIDAD DE CAPTURA': 'UNICAP'}, inplace=True)
df_filtrado_2017_2021.rename(columns={'DEPARTAMENTO': 'DEPTO'}, inplace=True)
df_filtrado_2017_2021.rename(columns={'NRO CORRESPONSALES PROPIOS': 'NRO_CORRESPONSALES_PROPIOS'}, inplace=True)
df_filtrado_2017_2021.rename(columns={'TIPO DE ENTIDAD': 'TIPO_ENTIDAD'}, inplace=True)


df_filtrado_2021_2024.rename(columns={'DESCRIP_UC': 'DEPTO'}, inplace=True)
df_filtrado_2021_2024.rename(columns={'(1) NRO_CORRESPONSALES_FÍSICOS_PROPIOS': 'NRO_CORRESPONSALES_PROPIOS'}, inplace=True)


df_filtrado_2017_2021.head()
df_filtrado_2021_2024.head()

df_filtrado_2017_2021.iloc[:, 2:6].head()

df_filtrado_2021_2024['TIPO'].unique()

df_filtrado_2017_2021.columns
df_filtrado_2021_2024.columns

df_filtrado_2017_2021 = df_filtrado_2017_2021[
                                                ['TIPO_ENTIDAD', 'CODIGO_ENTIDAD', 'NOMBRE_ENTIDAD', 'FECHA_CORTE',
                                                'UNICAP', 'DEPTO', 'MUNICIPIO', 'Inclusion_financiera_general',
                                                'Inclusion_financiera_men', 'Inclusion_financiera_women',
                                                'NRO_CORRESPONSALES_PROPIOS']
                                            ]

df_filtrado_2021_2024 = df_filtrado_2021_2024[
                                                ['TIPO_ENTIDAD', 'CODIGO_ENTIDAD', 'NOMBRE_ENTIDAD', 'FECHA_CORTE',
                                                'UNICAP', 'DEPTO', 'MUNICIPIO', 'Inclusion_financiera_general',
                                                'Inclusion_financiera_men', 'Inclusion_financiera_women',
                                                'NRO_CORRESPONSALES_PROPIOS']
                                            ]
# Concatenar verticalmente
df_concatenado = pd.concat([df_filtrado_2017_2021, df_filtrado_2021_2024], axis=0)

# Opcional: ordena por fecha si vas a hacer análisis temporal
df_concatenado['FECHA_CORTE'] = pd.to_datetime(df_concatenado['FECHA_CORTE'])

df_concatenado.to_csv(os.path.join(ruta_bases, "corresponsales_consolidado.csv"))

#df_concatenado = df_concatenado.sort_values('FECHA_CORTE').reset_index(drop=True)


#--------------------------------
serie_2017_2021 = df_filtrado_2017_2021.groupby('FECHA_CORTE')['NRO CORRESPONSALES PROPIOS'].sum()
serie_2021_2024 = df_filtrado_2021_2024.groupby('FECHA_CORTE')['(1) NRO_CORRESPONSALES_FÍSICOS_PROPIOS'].sum()
serie_total = pd.concat([serie_2017_2021, serie_2021_2024])
serie_total = serie_total.sort_index()  # Asegura el orden temporal
serie_total.index = pd.to_datetime(serie_total.index)
serie_total = serie_total.sort_index()

df_filtrado_2021_2024['NOMBRE_ENTIDAD'].nunique()

"""
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
serie_total.plot(marker='o')
plt.title('Evolución de Corresponsales Propios en Colombia')
plt.xlabel('Fecha de Corte')
plt.ylabel('Total Nacional')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ruta_resultados, "Evolución_Corresponsales_propias.png"), dpi=300)
plt.show()
#-------------------------------------------------------------------
# Asegúrate de que FECHA_CORTE esté en formato datetime
df_filtrado_2021_2024['FECHA_CORTE'] = pd.to_datetime(df_filtrado_2021_2024['FECHA_CORTE'])

# Filtra para quitar Bancolombia
df_sin_bancolombia = df_filtrado_2021_2024[df_filtrado_2021_2024['NOMBRE_ENTIDAD'] == 'Bancolombia']

# Agrupa y suma
serie_tercerizados = df_sin_bancolombia.groupby('FECHA_CORTE')['(80) NRO_CORRESPONSALES_FÍSICOS_TERCERIZADOS_ACTIVOS'].sum()

# Grafica
plt.figure(figsize=(10, 5))
serie_tercerizados.sort_index().plot(marker='o', color='green')
plt.title('Evolución de Corresponsales Físicos Tercerizados Activos (Bancolombia)')
plt.xlabel('Fecha de Corte')
plt.ylabel('Total Nacional')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ruta_resultados, "Evolucion_Corresponsales_Tercerizados_Bancolombia.png"), dpi=300)
plt.show()
#-----------------------------------------------------------------------
df_filtrado_2017_2021['NOMBRE_ENTIDAD'].unique()
# 1. Filtrar Bancolombia en ambos periodos
df_2017_2021_sin_bancolombia = df_filtrado_2017_2021[df_filtrado_2017_2021['NOMBRE_ENTIDAD'] != 'Bancolombia']
df_2021_2024_sin_bancolombia = df_filtrado_2021_2024[df_filtrado_2021_2024['NOMBRE_ENTIDAD'] != 'Bancolombia']

# 2. Agrupar por fecha
serie_2017_2021_sin_bancolombia = df_2017_2021_sin_bancolombia.groupby('FECHA_CORTE')['NRO CORRESPONSALES PROPIOS'].sum()
serie_2021_2024_sin_bancolombia = df_2021_2024_sin_bancolombia.groupby('FECHA_CORTE')['(1) NRO_CORRESPONSALES_FÍSICOS_PROPIOS'].sum()

# 3. Concatenar series
serie_total_sin_bancolombia = pd.concat([serie_2017_2021_sin_bancolombia, serie_2021_2024_sin_bancolombia])

# 4. Asegurar formato de fecha y orden
serie_total_sin_bancolombia.index = pd.to_datetime(serie_total_sin_bancolombia.index)
serie_total_sin_bancolombia = serie_total_sin_bancolombia.sort_index()

# 5. Graficar
plt.figure(figsize=(10, 5))
serie_total_sin_bancolombia.plot(marker='o', color='darkblue')
plt.title('Evolución de Corresponsales Propios en Colombia (Sin Bancolombia)')
plt.xlabel('Fecha de Corte')
plt.ylabel('Total Nacional')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# 6. Guardar
plt.savefig(os.path.join(ruta_resultados, "Evolucion_Corresponsales_Propios_Sin_Bancolombia.png"), dpi=300)
plt.show()
#---------------------------------------------------------
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Agrupar datos
df_interactivo = df_filtrado_2021_2024.groupby(['FECHA_CORTE', 'NOMBRE_ENTIDAD'])['(1) NRO_CORRESPONSALES_FÍSICOS_PROPIOS'].sum().reset_index()
df_interactivo['FECHA_CORTE'] = pd.to_datetime(df_interactivo['FECHA_CORTE'])

# Seleccionar top 5 entidades por total acumulado
top_entidades = df_interactivo.groupby('NOMBRE_ENTIDAD')['(1) NRO_CORRESPONSALES_FÍSICOS_PROPIOS'].sum().nlargest(51).index
df_top = df_interactivo[df_interactivo['NOMBRE_ENTIDAD'].isin(top_entidades)]

# Crear gráfica con Plotly
fig = px.line(df_top,
              x='FECHA_CORTE',
              y='(1) NRO_CORRESPONSALES_FÍSICOS_PROPIOS',
              color='NOMBRE_ENTIDAD',
              title='Corresponsales Físicos Propios (2021 a 2024) por entidad',
              markers=True)

# Fondo blanco
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis_title='Fecha de Corte',
    yaxis_title='Número de Corresponsales',
    legend_title='Entidad Financiera',
    hovermode='x unified'
)

# Línea vertical en 31/03/2021
fig.add_vline(x='2021-03-31', 
              line_dash='dash', 
              line_color='red', 
              annotation_text='Corte Mar-2021',
              annotation_position='top left')
fig.write_image(os.path.join(ruta_resultados, "Evolucion_Corresponsales_Propios_Entidades.png"), scale=3)
fig.show()

"""

#---------------------------------------------------------------------------------------
#            Ajuste de la base de datos cobertura_movil y internet_fijo 
#-----------------------------------
#Agrupamiento de municipios en común
#-----------------------------------
cobertura_movil.columns
internet_fijo.columns

cobertura_movil['FECHA_CORTE'] = pd.to_datetime(
    cobertura_movil['AÑO'].astype(str) + 'Q' + cobertura_movil['TRIMESTRE'].astype(str)
).dt.to_period('Q').dt.start_time

internet_fijo['FECHA_CORTE'] = pd.to_datetime(
    internet_fijo['AÑO'].astype(str) + 'Q' + internet_fijo['TRIMESTRE'].astype(str)
).dt.to_period('Q').dt.start_time

#   Ajuste paara las variables de cobertura movil      #
# Asegúrate de que las columnas de cobertura estén en formato numérico
columnas_cobertura = ['COBERTURA 2G', 'COBERTURA 3G', 'COBERTURA HSPA+, HSPA+DC',
                      'COBERTUTA 4G', 'COBERTURA LTE', 'COBERTURA 5G']

cobertura_movil[columnas_cobertura] = cobertura_movil[columnas_cobertura].replace({'S': 1, 'N': 0})

# Agrupamos por municipio y fecha
cobertura_agregada = cobertura_movil.groupby(
    ['COD MUNICIPIO', 'MUNICIPIO', 'FECHA_CORTE']
)[columnas_cobertura].mean().reset_index()


# Escala de 0 a 100
cobertura_agregada[columnas_cobertura] = cobertura_agregada[columnas_cobertura] * 100

#municipios_b = [m for m in cobertura_agregada['MUNICIPIO'].unique() if str(m).startswith('FU')]
#cobertura_agregada[cobertura_agregada['MUNICIPIO']=='FUSAGASUGÁ'].sort_values(by='FECHA_CORTE')

internet_fijo['MUNICIPIO'].unique().shape
cobertura_agregada['MUNICIPIO'].unique().shape

internet_fijo.rename(columns={'INDICE': 'INDICE_INTERNET_FIJO'}, inplace=True)
internet_fijo = internet_fijo[[ 'DEPARTAMENTO', 'COD_MUNICIPIO',
       'MUNICIPIO', 'POBLACIÓN DANE',
       'INDICE_INTERNET_FIJO', 'FECHA_CORTE']]

internet_fijo.columns
cobertura_agregada.columns

cobertura_agregada = pd.merge(
    internet_fijo,
    cobertura_agregada,
    on=['MUNICIPIO', 'FECHA_CORTE'],
    how='outer',
    suffixes=('_INTERNET', '_COBERTURA')
)

"""
#           Revisión de que esten totalmente completos
# Normalizar los nombres de municipios para comparación
municipios_fijo = internet_fijo['MUNICIPIO'].str.strip().str.upper().unique()
municipios_cobertura = cobertura_agregada['MUNICIPIO'].str.strip().str.upper().unique()

# Encontrar diferencias
solo_en_fijo = set(municipios_fijo) - set(municipios_cobertura)
solo_en_cobertura = set(municipios_cobertura) - set(municipios_fijo)

print("Municipios que están SOLO en internet_fijo:")
print(sorted(solo_en_fijo))

print("\nMunicipios que están SOLO en cobertura_agregada:")
print(sorted(solo_en_cobertura))

"""

# En df_concatenado
df_concatenado['FECHA_CORTE'] = pd.to_datetime(df_concatenado['FECHA_CORTE'], dayfirst=True, errors='coerce')
df_concatenado['TRIMESTRE'] = df_concatenado['FECHA_CORTE'].dt.to_period('Q')

# En cobertura_agregada
cobertura_agregada['FECHA_CORTE'] = pd.to_datetime(cobertura_agregada['FECHA_CORTE'], errors='coerce')
cobertura_agregada['TRIMESTRE'] = cobertura_agregada['FECHA_CORTE'].dt.to_period('Q')



# -----------------------------
# 1. Lista de palabras que indican que NO es un municipio
# -----------------------------
palabras_no_municipios = [
    "ACTIVIDADES", "DEPÓSITO", "DEPÓSITOS", "GIROS", "COMERCIO", "SERVICIOS", "NUM", 
    "CORRESPONSALES", "OTROS", "RED", "PAGOS", "ESTABLECIMIENTOS", "FULLCARGA", "EFECTY",
    "SUPERGIROS", "PAGA TODO", "SERVYPAGOS", "BQUANTUM", "ASSENDA", "DISTRACOM", "DISTRICOL",
    "EDEQ", "E-PAGO", "MERCAR", "COMCARD"
]

# -----------------------------
# 2. Funciones auxiliares
# -----------------------------
def limpiar_nombre(nombre):
    if pd.isna(nombre):
        return ''
    nombre = str(nombre).upper()
    nombre = re.sub(r'\(.*?\)', '', nombre)
    nombre = re.sub(r'[^A-Z\s]', '', nombre)
    nombre = unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('utf-8')
    nombre = re.sub(r'\s+', ' ', nombre).strip()
    return nombre

def filtrar_municipios_validos(lista_municipios):
    return [m for m in lista_municipios if all(palabra not in m for palabra in palabras_no_municipios)]

def normalizar(municipios):
    return [unidecode.unidecode(m).replace('  ', ' ').strip().upper() for m in municipios]

# -----------------------------
# 3. Limpiar y normalizar columnas de municipios
# -----------------------------
df_concatenado["Municipio_limpio"] = df_concatenado["MUNICIPIO"].apply(limpiar_nombre)
cobertura_agregada["Municipio_limpio"] = cobertura_agregada["MUNICIPIO"].apply(limpiar_nombre)

# -----------------------------
# 4. Obtener listas únicas
# -----------------------------
municipios_df = df_concatenado["Municipio_limpio"].dropna().unique()
municipios_cob = cobertura_agregada["Municipio_limpio"].dropna().unique()

# -----------------------------
# 5. Filtrar nombres no válidos y normalizar
# -----------------------------
municipios_df = filtrar_municipios_validos(municipios_df)
municipios_cob = filtrar_municipios_validos(municipios_cob)
municipios_df = normalizar(municipios_df)
municipios_cob = normalizar(municipios_cob)

# -----------------------------
# 6. Comparar diferencias
# -----------------------------
solo_en_df = sorted(set(municipios_df) - set(municipios_cob))

# -----------------------------
# 7. Corrección automática con similitud
# -----------------------------
nombres_malos = municipios_cob
nombres_correctos = solo_en_df
umbral_similitud = 80

# Normalizar referencia para matching
nombres_correctos_norm = normalizar([limpiar_nombre(m) for m in nombres_correctos])
nombre_original_map = dict(zip(nombres_correctos_norm, nombres_correctos))

sugerencias = {}

for nombre in nombres_malos:
    mejor_match, score, _ = process.extractOne(nombre, nombres_correctos_norm, scorer=fuzz.token_sort_ratio)
    if score >= umbral_similitud and nombre != mejor_match:
        sugerencias[nombre] = nombre_original_map[mejor_match]
    else:
        sugerencias[nombre] = None

# -----------------------------
# 8. Filtrar y aplicar correcciones
# -----------------------------
correcciones_utiles = {k: v for k, v in sugerencias.items() if v and k != v}

print(f"\nTotal de municipios con corrección sugerida: {len(correcciones_utiles)}\n")
for original, sugerido in correcciones_utiles.items():
    print(f"{original} --> {sugerido}")

# Aplicar corrección sobre el DataFrame original (opcional: puedes hacerlo en ambos)
df_concatenado["Municipio_limpio_corregido"] = df_concatenado["Municipio_limpio"].apply(
    lambda x: correcciones_utiles.get(x, x)
)
cobertura_agregada["Municipio_limpio_corregido"] = cobertura_agregada["Municipio_limpio"].apply(
    lambda x: correcciones_utiles.get(x, x)
)

#--------------------------------------------
#-----------

municipios_df = df_concatenado["Municipio_limpio_corregido"].dropna().unique()
municipios_cob = cobertura_agregada["Municipio_limpio_corregido"].dropna().unique()

municipios_df = filtrar_municipios_validos(municipios_df)
municipios_cob = filtrar_municipios_validos(municipios_cob)

municipios_df = normalizar(municipios_df)
municipios_cob = normalizar(municipios_cob)
# Convertir a sets
set_df = set(municipios_df)
set_cob = set(municipios_cob)

# Municipios que están en df_concatenado pero no en cobertura_agregada
solo_en_df = sorted(set_df - set_cob)

# Municipios que están en cobertura_agregada pero no en df_concatenado
solo_en_cob = sorted(set_cob - set_df)

# Mostrar resultados
print("Municipios solo en df_concatenado:", len(solo_en_df))
print(solo_en_df)

print("\nMunicipios solo en cobertura_agregada:", len(solo_en_cob))
print(solo_en_cob)

#---------------------------

# Hacer merge tipo outer
df_with_cobertura = pd.merge(
    df_concatenado,
    cobertura_agregada,
    on=['Municipio_limpio_corregido', 'TRIMESTRE'],
    how='outer',
    suffixes=('_INTERNET', '_COBERTURA')
)
df_with_cobertura.shape

df_with_cobertura['Municipio_limpio_COBERTURA'].isna().sum()
df_with_cobertura['Municipio_limpio_INTERNET'].isna().sum()

df_with_cobertura = df_with_cobertura.dropna(subset=['Municipio_limpio_COBERTURA', 'Municipio_limpio_INTERNET'])
df_with_cobertura.shape

df_with_cobertura.isna().sum()
df_with_cobertura = df_with_cobertura.drop(columns=['Municipio_limpio_COBERTURA', 'Municipio_limpio_INTERNET'])

df_with_cobertura.columns

df_with_cobertura.isna().sum()
# Eliminar filas con al menos un NaN
df_with_cobertura = df_with_cobertura.dropna()


# Guardar el DataFrame limpio en un archivo CSV
df_with_cobertura.to_csv(os.path.join(ruta_bases, "df_with_cobertura_limpio.csv"), index=False)

# Asegúrate de no dividir por cero ni por valores nulos
df_with_cobertura = df_with_cobertura[df_with_cobertura["POBLACIÓN DANE"].notna()]
df_with_cobertura = df_with_cobertura[df_with_cobertura["POBLACIÓN DANE"] != 0]

# División manual
df_with_cobertura["Inclusion_financiera_general_relativa"] = (
    df_with_cobertura["Inclusion_financiera_general"] / df_with_cobertura["POBLACIÓN DANE"]
)

df_with_cobertura["Inclusion_financiera_men_relativa"] = (
    df_with_cobertura["Inclusion_financiera_men"] / df_with_cobertura["POBLACIÓN DANE"]
)

df_with_cobertura["Inclusion_financiera_women_relativa"] = (
    df_with_cobertura["Inclusion_financiera_women"] / df_with_cobertura["POBLACIÓN DANE"]
)

#----------------------------
df_with_cobertura[[
    "Inclusion_financiera_general_relativa",
    "Inclusion_financiera_men_relativa",
    "Inclusion_financiera_women_relativa"
]].notna().sum()

(df_with_cobertura[[
    "Inclusion_financiera_general_relativa",
    "Inclusion_financiera_men_relativa",
    "Inclusion_financiera_women_relativa"
]] >= 0).sum()
#-------------------------------

df_with_cobertura.shape
df_with_cobertura.columns
df_with_cobertura.head()


# Asegúrate de que COD MUNICIPIO y FECHA_CORTE_COBERTURA sean strings o categorías
df_with_cobertura['COD MUNICIPIO'] = df_with_cobertura['COD MUNICIPIO'].astype(str)
df_with_cobertura['FECHA_CORTE_COBERTURA'] = pd.to_datetime(df_with_cobertura['FECHA_CORTE_COBERTURA'])
