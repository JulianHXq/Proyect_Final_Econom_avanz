#---------------------------------------------------------------------------------------
#          Planteamiento del Modelo DID-Propensity score matching (Event study)
#                               Econometría avanzada
#---------------------------------------------------------------------------------------

#Load libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define de paths
ruta_bases = r"C:\Users\User\Desktop\EconoAvanz\Trabajo_Final\bases" 
ruta_resultados = r"C:\Users\User\Desktop\EconoAvanz\Trabajo_Final\resultados" 

#Load Data base clear (Clear_data.py)
df_load = pd.read_csv(os.path.join(ruta_bases, "df_with_cobertura_limpio.csv"))

# Copia del DataFrame
df = df_load.copy()


# --- 1. Crear columna de semestre ---
df['FECHA_CORTE_COBERTURA'] = pd.to_datetime(df['FECHA_CORTE_COBERTURA'])
df['SEMESTRE'] = df['FECHA_CORTE_COBERTURA'].dt.month.map(lambda x: 1 if x <= 6 else 2)
df['AÑO'] = df['FECHA_CORTE_COBERTURA'].dt.year
df['PERIODO_SEMESTRAL'] = df['AÑO'].astype(str) + 'S' + df['SEMESTRE'].astype(str)

# --- 2. Convertir a numérico si hay comas ---
for col in ['NRO_CORRESPONSALES_PROPIOS', 'INDICE_INTERNET_FIJO', 'COBERTURA 3G', 'COBERTUTA 4G']:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

df = df.rename(columns={'COBERTUTA 4G': 'COBERTURA_4G'})

# --- 3. Agregación semestral ---
df_sem = df.groupby([
    'MUNICIPIO_INTERNET', 'PERIODO_SEMESTRAL', 'COD_MUNICIPIO', 'MUNICIPIO_COBERTURA'
], as_index=False).agg({
    'Inclusion_financiera_general_relativa': 'sum',
    'NRO_CORRESPONSALES_PROPIOS': 'sum',
    'INDICE_INTERNET_FIJO': 'mean',
    'COBERTURA 3G': 'mean',
    'POBLACIÓN DANE': 'mean',
    'COBERTURA 2G': 'mean',
    'COBERTURA HSPA+, HSPA+DC': 'mean',
    'COBERTURA LTE': 'mean',
    'COBERTURA 5G': 'mean',
    'Inclusion_financiera_men_relativa': 'sum',
    'Inclusion_financiera_women_relativa': 'sum',
    'COBERTURA_4G': 'mean'
})
df_sem.rename(columns={
    'COBERTURA 3G': 'COBERTURA_3G',
    'POBLACIÓN DANE': 'POBLACION_DANE',
    'COBERTURA 2G': 'COBERTURA_2G',
    'COBERTURA HSPA+, HSPA+DC': 'COBERTURA_HSPA_PLUS',
    'COBERTURA LTE': 'COBERTURA_LTE',
    'COBERTURA 5G': 'COBERTURA_5G',
    'COBERTURA 4G': 'COBERTURA_4G'
}, inplace=True)

variables_cobertura = [
    'INDICE_INTERNET_FIJO', 
    'COBERTURA_3G',
    'COBERTURA_4G',
    'COBERTURA_LTE',
]
# --- 3.1 Estandarizar
scaler = StandardScaler()
X_cobertura = scaler.fit_transform(df_sem[variables_cobertura])

# --- 3.2 PCA: aquí extraemos 1 o más componentes
pca = PCA(n_components=1)  # puedes cambiar a n_components=1 si quieres solo una dimensión resumen
componentes = pca.fit_transform(X_cobertura)

# --- 3.3 Agregar los componentes al DataFrame
df_sem['PCA_COBERTURA'] = componentes[:, 0]

# Coeficientes de cada variable en el componente principal
pca_result = pd.DataFrame(
    pca.components_,
    columns=variables_cobertura,
    index=['PCA']
)
print(pca_result)


df_matched = df_sem.copy()
#df_matched['PERIODO_SEMESTRAL'].unique()
df_matched = df_matched[~df_matched['PERIODO_SEMESTRAL'].isin(['2017S2', '2018S1'])]
df_matched = df_matched[df_matched['Inclusion_financiera_general_relativa'] < 6]
# ----------------------------------------------------
# 4. Crear la variable de tratamiento
# ----------------------------------------------------
# Tratados: municipios que después de 2021S2 tienen más corresponsales
df_matched['delta_corresponsales'] = df_matched.groupby('COD_MUNICIPIO')['NRO_CORRESPONSALES_PROPIOS'].transform(lambda x: x.loc[x.index[-1]] - x.loc[x.index[0]])

umbral = df_matched['delta_corresponsales'].quantile(0.70)
df_matched['tratado'] = (df_matched['delta_corresponsales'] > umbral).astype(int)

# ----------------------------------------------------
# 5. Calcular propensity scores con covariables previas
# ----------------------------------------------------
# Escoger una sola observación por municipio antes del tratamiento
pre_tratamiento = df_matched[df_matched['PERIODO_SEMESTRAL'] <= '2021S2'].drop_duplicates('COD_MUNICIPIO')
X = pre_tratamiento[['PCA_COBERTURA', 'Inclusion_financiera_general_relativa']]
y = pre_tratamiento['tratado']

modelo_ps = LogisticRegression().fit(X, y)
pre_tratamiento['propensity_score'] = modelo_ps.predict_proba(X)[:,1]

# ----------------------------------------------------
# 6. Matching 1 a 1 con Nearest Neighbor
# ----------------------------------------------------
tratados = pre_tratamiento[pre_tratamiento['tratado'] == 1]
controles = pre_tratamiento[pre_tratamiento['tratado'] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(controles[['propensity_score']])
distancias, indices = nn.kneighbors(tratados[['propensity_score']])

controles_emparejados = controles.iloc[indices.flatten()].copy()
tratados_emparejados = tratados.reset_index(drop=True)

municipios_emparejados = pd.concat([tratados_emparejados, controles_emparejados])['COD_MUNICIPIO'].unique()
df_matched = df_matched[df_matched['COD_MUNICIPIO'].isin(municipios_emparejados)]

# ----------------------------------------------------
# 7. Construir variables de evento
# ----------------------------------------------------

# Reutiliza tus interacciones: NRO_CORRESPONSALES_PROPIOS * periodo
dummies_periodo = pd.get_dummies(df_matched['PERIODO_SEMESTRAL'], prefix='time')
df_matched = pd.concat([df_matched, dummies_periodo], axis=1)

duplicadas = df_matched.columns[df_matched.columns.duplicated()].tolist()
print("Columnas duplicadas:", duplicadas)

# 1. Eliminar columnas duplicadas
df_matched = df_matched.loc[:, ~df_matched.columns.duplicated()]

# 2. Verificar nuevamente si quedó limpio
print("¿Duplicadas aún?", df_matched.columns.duplicated().any())

# 3. (Opcional) Eliminar dummies existentes si vas a reconstruirlos
dummies_existentes = [col for col in df_matched.columns if col.startswith('time_')]
df_matched.drop(columns=dummies_existentes, inplace=True)
df_matched = df_matched[df_matched['PERIODO_SEMESTRAL'] !='2017S2']

# 4. Volver a generar las dummies de periodo
dummies_periodo = pd.get_dummies(df_matched['PERIODO_SEMESTRAL'], prefix='time')

# 5. Añadir las dummies (ya sin duplicadas)
df_matched = pd.concat([df_matched, dummies_periodo], axis=1)

# 6. Crear interacciones: oficinas * dummies de periodo
for col in dummies_periodo.columns:
    df_matched[f'corresp_{col}'] = df_matched['NRO_CORRESPONSALES_PROPIOS'] * df_matched[col]

# Tendencia específica por municipio
df_matched['tiempo_numerico'] = df_matched['PERIODO_SEMESTRAL'].astype('category').cat.codes
df_matched['trend_municipio'] = df_matched['tiempo_numerico'] * df_matched['COD_MUNICIPIO'].astype('category').cat.codes


# Dummies de tiempo
dummies = [col for col in dummies_periodo.columns if col != 'time_2021S2']
interacciones = [f'corresp_{col}' for col in dummies]

# Interacción post
df_matched['post'] = (df_matched['PERIODO_SEMESTRAL'] >= '2021S2').astype(int)
df_matched['corresp_post'] = df_matched['NRO_CORRESPONSALES_PROPIOS'] * df_matched['post']

# ----------------------------------------------------
# 8. Estimar el modelo de eventos con DID
# ----------------------------------------------------

formula = 'Inclusion_financiera_general_relativa ~ ' + \
          ' + '.join(dummies) + ' + ' + \
          ' + '.join(interacciones) + ' + ' + \
          'corresp_post   + PCA_COBERTURA + trend_municipio + C(MUNICIPIO_COBERTURA)'

df_matched.columns
modelo_evento = smf.ols(formula=formula, data=df_matched).fit()
print(modelo_evento.summary())
with open(r'C:\Users\User\Desktop\EconoAvanz\Trabajo_Final\resultados\resumen_modelo_evento.txt','w') as f:
    f.write(modelo_evento.summary().as_text())
# ----------------------------------------------------
# 9. Graficar el evento (efecto dinámico)
# ----------------------------------------------------
coeficientes = modelo_evento.params[interacciones]
errores = modelo_evento.bse[interacciones]
periodos = [col.replace('corresp_time_', '') for col in interacciones]

plt.figure(figsize=(10, 6))
plt.errorbar(periodos, coeficientes, yerr=1.96 * errores, fmt='o', capsize=4, color='blue')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(x=periodos.index('2021S1'), color='black', linestyle='--', label='Tratamiento')
plt.xticks(rotation=45)
plt.title('Efecto marginal de corresponsales por semestre (event study)')
plt.xlabel('Semestre')
plt.ylabel('Efecto estimado sobre inclusión financiera')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ruta_resultados, "DID_efecto_dinamico.png"), dpi=300)
plt.show()

# ----------------------------------------------------
# 10. Comprobacion de tendencias paralelas
# ---------------------------------------------------
                         
# Agrupar por tiempo y grupo
df_avg = df_matched.groupby(['PERIODO_SEMESTRAL', 'tratado'])['Inclusion_financiera_general_relativa'].mean().reset_index()

#df_matched[df_matched['Inclusion_financiera_general_relativa'] > 5][['PERIODO_SEMESTRAL', 'tratado', 'Inclusion_financiera_general_relativa']]

# Graficar
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_avg, x='PERIODO_SEMESTRAL', y='Inclusion_financiera_general_relativa', hue='tratado')
plt.axvline(x='2021S2', color='gray', linestyle='--', label='Tratamiento')
plt.title('Tendencias en inclusión financiera: tratados vs. no tratados')
plt.xlabel('Periodo')
plt.ylabel('Inclusión financiera')
plt.legend(title='Tratado')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ruta_resultados, "Tendencias_paralelas.png"), dpi=300)
plt.show()

# Filtrar solo periodos antes del tratamiento
df_pre = df_matched[df_matched['PERIODO_SEMESTRAL'] < '2021S2'].copy()

# Crear una variable de tendencia temporal (números enteros)
df_pre['trend_time'] = df_pre['PERIODO_SEMESTRAL'].astype('category').cat.codes

# Interacción entre tratado y tendencia temporal
df_pre['tratado_trend'] = df_pre['trend_time'] * df_pre['tratado']

# Regresión para testear tendencias paralelas
modelo_pp = smf.ols('Inclusion_financiera_general_relativa ~ trend_time + PCA_COBERTURA + tratado + tratado_trend', data=df_pre).fit()
print(modelo_pp.summary())
with open(r'C:\Users\User\Desktop\EconoAvanz\Trabajo_Final\resultados\resumen_modelo_pp.txt','w') as f:
    f.write(modelo_pp.summary().as_text())
# ----------------------------------------------------
# 11. Revisión de los efecto por genero
# ---------------------------------------------------
formula_men = 'Inclusion_financiera_men_relativa ~ ' + \
                 ' + '.join(dummies) + ' + ' + \
                 ' + '.join(interacciones) + ' + ' + \
                 'corresp_post  + PCA_COBERTURA  + trend_municipio + C(MUNICIPIO_COBERTURA)'

modelo_evento_men = smf.ols(formula=formula_men, data=df_matched).fit()

formula_women = 'Inclusion_financiera_women_relativa ~ ' + \
                 ' + '.join(dummies) + ' + ' + \
                 ' + '.join(interacciones) + ' + ' + \
                 'corresp_post  + PCA_COBERTURA  + trend_municipio + C(MUNICIPIO_COBERTURA)'


modelo_evento_women  = smf.ols(formula=formula_women, data=df_matched).fit()

#Revision de los modelos
print(modelo_evento_men.summary())
print(modelo_evento_women.summary())

with open(r'C:\Users\User\Desktop\EconoAvanz\Trabajo_Final\resultados\resumen_modelo_men.txt','w') as f:
    f.write(modelo_evento_men.summary().as_text())
with open(r'C:\Users\User\Desktop\EconoAvanz\Trabajo_Final\resultados\resumen_modelo_women.txt','w') as f:
    f.write(modelo_evento_women.summary().as_text())

# ----------------------------------------------------
# 12. Graficas del efecto por genero
# ---------------------------------------------------
# Coeficientes y errores estándar para ambas versiones
coef_con_men = modelo_evento_men.params[interacciones]
error_con_men = modelo_evento_men.bse[interacciones]

coef_sin_women = modelo_evento_women.params[interacciones]
error_sin_women = modelo_evento_women.bse[interacciones]

# Eje x
periodos = [col.replace('corresp_time_', '') for col in interacciones]
x = list(range(len(periodos)))

# Graficar
plt.figure(figsize=(12, 6))

# Con hombres
plt.errorbar(x, coef_con_men, yerr=1.96 * error_sin_women, fmt='o', capsize=4,
             color='blue', label='Hombres')

# Con mujeres
plt.errorbar(x, coef_sin_women, yerr=1.96 * error_sin_women, fmt='s', capsize=4,
             color='orange', label='Mujerees')

# Líneas de referencia
plt.axhline(0, color='gray', linestyle='--')
if '2021S1' in periodos:
    plt.axvline(x=periodos.index('2021S1'), color='black', linestyle='--', label='Tratamiento (2021S1)')

# Estética
plt.xticks(ticks=x, labels=periodos, rotation=45)
plt.xlabel('Semestre')
plt.ylabel('Efecto estimado')
plt.title('Comparación del efecto de corresponsales en el tiempo\n(por genero)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ruta_resultados, "DID_efecto_dinamico_genero.png"), dpi=300)
plt.show()

# ----------------------------------------------------
# 13. Prueba de placebo
# ---------------------------------------------------
                                      
# Crear variable placebo como si el tratamiento hubiera sido en 2020S1
df_placebo = df_matched.copy()
df_placebo['placebo_post'] = (df_placebo['PERIODO_SEMESTRAL'] >= '2020S1').astype(int)
df_placebo['placebo_tratado'] = df_placebo['tratado']
df_placebo['placebo_post_tratado'] = df_placebo['placebo_post'] * df_placebo['placebo_tratado']

# Usamos solo datos antes del tratamiento real (2021S2)
df_placebo = df_placebo[df_placebo['PERIODO_SEMESTRAL'] < '2021S2']


modelo_placebo = smf.ols('Inclusion_financiera_general_relativa ~ placebo_post + placebo_tratado + placebo_post_tratado + PCA_COBERTURA + trend_municipio + C(MUNICIPIO_COBERTURA)', data=df_placebo).fit()
print(modelo_placebo.summary())


with open(r'C:\Users\User\Desktop\EconoAvanz\Trabajo_Final\resultados\resumen_modelo_placebo.txt','w') as f:
    f.write(modelo_placebo.summary().as_text())

# ----------------------------------------------------
# 13. Prueba de balance
# ---------------------------------------------------
from scipy.stats import ttest_ind

# Filtrar datos solo hasta el semestre 2021S2 (pretratamiento)
df_pretratamiento = df_matched[df_matched['PERIODO_SEMESTRAL'] <= '2021S2'].copy()

variables = [
    'INDICE_INTERNET_FIJO',
    'COBERTURA_3G',
    'COBERTURA_2G',
    'COBERTURA_HSPA_PLUS',
    'COBERTURA_LTE',
    'COBERTURA_4G',
    'PCA_COBERTURA',
]
balance_results = []

for var in variables:
    treated_vals = df_pretratamiento[df_pretratamiento['tratado'] == 1][var].dropna()
    control_vals = df_pretratamiento[df_pretratamiento['tratado'] == 0][var].dropna()
    
    mean_treated = treated_vals.mean()
    mean_control = control_vals.mean()
    std_treated = treated_vals.std()
    std_control = control_vals.std()
    
    # t-test
    t_stat, p_val = ttest_ind(treated_vals, control_vals, equal_var=False)
    
    balance_results.append({
        'Variable': var,
        'Media Tratado': mean_treated,
        'Media Control': mean_control,
        'Std Tratado': std_treated,
        'Std Control': std_control,
        'p-value': p_val
    })
print(pca_result)

# Mostrar como DataFrame ordenado
import pandas as pd
pd.DataFrame(balance_results).round(3)

#------------------------------------------------------


# Lista de códigos DANE de capitales departamentales
capitales_dane = [
    91001,  # Leticia
    5001,  # Medellín
    81001,  # Arauca
    8001,  # Barranquilla
    11001,  # Bogotá
    13001,  # Cartagena
    15001,  # Tunja
    17001,  # Manizales
    18001,  # Florencia
    85001,  # Yopal
    19001,  # Popayán
    20001,  # Valledupar
    27001,  # Quibdó
    23001,  # Montería
    94001,  # Inírida
    95001,  # San José del Guaviare
    41001,  # Neiva
    44001,  # Riohacha
    47001,  # Santa Marta
    50001,  # Villavicencio
    52001,  # Pasto
    54001,  # Cúcuta
    86001,  # Mocoa
    63001,  # Armenia
    66001,  # Pereira
    88001,  # San Andrés
    68001,  # Bucaramanga
    70001,  # Sincelejo
    73001,  # Ibagué
    76001,  # Cali
    97001,  # Mitú
    99001   # Puerto Carreño
]
