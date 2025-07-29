import pandas as pd
import seaborn as sb
import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Titulo para que salga en la página web
st.title("Predicción de Aprobación de Estudiantes con Árbol de Decisión")
st.markdown("Este modelo usa notas: Parciales, Proyecto y Examen Final para predecir si un estudiante aprobaría la materia")

# Cargar los datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("estudiantes_notas_finales.csv")

df = cargar_datos()

# Mostrar los primeros datos
st.subheader("Datos cargados")
st.write(df.head())

# Gráficos simples
st.subheader("Distribución de notas")
st.bar_chart(df[["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final", "Nota_Final"]].mean())

# Dividir las variables
x = df[["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final"]]
y = df["Aprobado"]

# Preparar los conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Entrenar modelo
modelo = DecisionTreeClassifier(max_depth=4, random_state=0)
modelo.fit(x_train, y_train)

# Evaluación del modelo
y_pred = modelo.predict(x_test)
st.subheader("Evaluación del Modelo")
st.text(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")

# Visualizar el árbol
st.subheader("Visualización del Árbol")
fig, ax = plt.subplots(figsize=(12,6))
plot_tree(modelo, feature_names=x.columns, class_names=["NO", "SI"], filled=True, rounded=True, fontsize=10)
st.pyplot(fig)

# Interacción
st.subheader("¿Aprobaría este Estudiante?")
with st.form("Formulario de Predicción de Notas"):
    p1 = st.number_input("Primer Parcial ", 0.0, 100.0, 50.0)
    p2 = st.number_input("Segundo Parcial ", 0.0, 100.0, 50.0)
    proy = st.number_input("Proyecto ", 0.0, 100.0, 50.0)
    ef = st.number_input("Examen Final ", 0.0, 100.0, 50.0)
    submitted = st.form_submit_button("Predecir")

if submitted:
    datos_nuevos = pd.DataFrame([[p1, p2, proy, ef]], columns=["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final"])
    prediccion = modelo.predict(datos_nuevos)[0]
    st.success(f"Resultado: {'Aprobado' if prediccion == 1 else 'Reprobado'}")
'''

# Sobrescribir archivos con la versión corregida
with open("/mnt/data/app.py", "w", encoding="utf-8") as f:
    f.write(app_code_fixed)

# El archivo requirements.txt no necesita cambios, se mantiene igual


Analizado
python
Mostrar siempre los detalles

Copiar
# Reejecutar tras el reinicio del entorno para regenerar los archivos

# Código corregido final con columnas explícitas en la predicción
app_code_fixed = '''\
import pandas as pd
import seaborn as sb
import streamlit as st 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Titulo para que salga en la página web
st.title("Predicción de Aprobación de Estudiantes con Árbol de Decisión")
st.markdown("Este modelo usa notas: Parciales, Proyecto y Examen Final para predecir si un estudiante aprobaría la materia")

# Cargar los datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("estudiantes_notas_finales.csv")

df = cargar_datos()

# Mostrar los primeros datos
st.subheader("Datos cargados")
st.write(df.head())

# Gráficos simples
st.subheader("Distribución de notas")
st.bar_chart(df[["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final", "Nota_Final"]].mean())

# Dividir las variables
x = df[["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final"]]
y = df["Aprobado"]

# Preparar los conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Entrenar modelo
modelo = DecisionTreeClassifier(max_depth=4, random_state=0)
modelo.fit(x_train, y_train)

# Evaluación del modelo
y_pred = modelo.predict(x_test)
st.subheader("Evaluación del Modelo")
st.text(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")

# Visualizar el árbol
st.subheader("Visualización del Árbol")
fig, ax = plt.subplots(figsize=(12,6))
plot_tree(modelo, feature_names=x.columns, class_names=["NO", "SI"], filled=True, rounded=True, fontsize=10)
st.pyplot(fig)

# Interacción
st.subheader("¿Aprobaría este Estudiante?")
with st.form("Formulario de Predicción de Notas"):
    p1 = st.number_input("Primer Parcial ", 0.0, 100.0, 50.0)
    p2 = st.number_input("Segundo Parcial ", 0.0, 100.0, 50.0)
    proy = st.number_input("Proyecto ", 0.0, 100.0, 50.0)
    ef = st.number_input("Examen Final ", 0.0, 100.0, 50.0)
    submitted = st.form_submit_button("Predecir")

if submitted:
    datos_nuevos = pd.DataFrame([[p1, p2, proy, ef]], columns=["Primer_Parcial", "Segundo_Parcial", "Proyecto", "Examen_Final"])
    prediccion = modelo.predict(datos_nuevos)[0]
    st.success(f"Resultado: {'Aprobado' if prediccion == 1 else 'Reprobado'}")
'''

# Requisitos para requirements.txt
requirements = '''\
streamlit
pandas
matplotlib
seaborn
scikit-learn
'''

# Guardar los archivos
with open("/mnt/data/app.py", "w", encoding="utf-8") as f:
    f.write(app_code_fixed)