import streamlit as st
import os
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------
# MODELOS
# -----------------------------
@st.cache_resource
def cargar_modelos():
    nlp = spacy.load("es_core_news_sm")
    @st.cache_resource
def cargar_modelo():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = cargar_modelo()
    return nlp, model

nlp, model = cargar_modelos()

# -----------------------------
# UI
# -----------------------------
st.title("📚 Clasificador de Cuentos")

st.sidebar.header("Configuración")

tema_input = st.sidebar.text_area(
    "Define el tema (una frase por línea)",
    "cobardía\nmiedo a actuar\nevitar decisiones"
)

tema_lista = [t.strip() for t in tema_input.split("\n") if t.strip()]

# -----------------------------
# VECTOR TEMA
# -----------------------------
tema_vector = model.encode(tema_lista, convert_to_tensor=True).mean(dim=0)

# -----------------------------
# FUNCIONES
# -----------------------------
def analizar(texto):

    doc = nlp(texto)

    palabras = [t.text.lower() for t in doc if t.is_alpha]
    total = len(palabras)

    oraciones = list(doc.sents)
    longitudes = [len([t for t in s if t.is_alpha]) for s in oraciones]
    avg_len = np.mean(longitudes) if longitudes else 0

    ttr = len(set(palabras)) / total if total else 0

    verbs = [t for t in doc if t.pos_ == "VERB"]
    nouns = [t for t in doc if t.pos_ == "NOUN"]
    adjs = [t for t in doc if t.pos_ == "ADJ"]

    verb_ratio = len(verbs)/total if total else 0
    densidad = (len(verbs)+len(nouns)+len(adjs))/total if total else 0

    emb = model.encode(texto, convert_to_tensor=True)
    sim = float(util.cos_sim(emb, tema_vector))

    return avg_len, ttr, verb_ratio, densidad, sim

# -----------------------------
# SUBIR ARCHIVOS
# -----------------------------
files = st.file_uploader(
    "Sube tus cuentos (.txt)",
    accept_multiple_files=True
)

if files:

    data = []

    for file in files:
        texto = file.read().decode("utf-8")
        avg_len, ttr, verb_ratio, densidad, sim = analizar(texto)

        data.append({
            "cuento": file.name,
            "longitud": avg_len,
            "ttr": ttr,
            "verb_ratio": verb_ratio,
            "densidad": densidad,
            "tema": sim
        })

    df = pd.DataFrame(data)

    # Escalar
    for col in ["longitud","ttr","verb_ratio","densidad","tema"]:
        df[col+"_esc"] = (
            1 + 4*(df[col] - df[col].min()) / (df[col].max() - df[col].min())
        ).fillna(3)

    # Score
    df["score"] = (
        df["longitud_esc"]*0.10 +
        df["ttr_esc"]*0.10 +
        df["densidad_esc"]*0.15 +
        df["tema_esc"]*0.25 +
        df["verb_ratio_esc"]*0.15
    )

    # Clusters
    features = df[["longitud_esc","ttr_esc","verb_ratio_esc","densidad_esc","tema_esc"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(features)

    st.subheader("Resultados")
    st.dataframe(df)

    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(features)

    df["x"] = coords[:,0]
    df["y"] = coords[:,1]

    # Gráfico
    fig, ax = plt.subplots()

    for _, row in df.iterrows():
        ax.scatter(row["x"], row["y"])
        ax.text(row["x"], row["y"], row["cuento"], fontsize=8)

    st.pyplot(fig)
