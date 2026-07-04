from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.database import get_recent_runs, get_stats

st.set_page_config(
    page_title="Historique: Assistant Radiologue",
    page_icon="📊",
    layout="wide",
)

st.markdown("## 📊 Historique des prédictions")
st.markdown("Toutes les analyses effectuées via l'application, stockées en base SQLite.")

stats = get_stats()

if not stats:
    st.info("Aucune prédiction enregistrée pour l'instant. Lancez une analyse depuis la page principale.")
    st.stop()

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
dist = stats.get("distribution", {})

col1.metric("Total analyses", stats["total"])
col2.metric("Confiance moyenne", f"{stats['avg_confidence']*100:.1f} %")
col3.metric("Latence moyenne", f"{stats['avg_latency_ms']:.0f} ms")
col4.metric("Taux incertitude",
            f"{dist.get('uncertain', 0) / max(stats['total'], 1) * 100:.1f} %")

st.markdown("---")

# --- Distribution des classes ---
col_chart, col_table = st.columns([1, 2])

with col_chart:
    st.markdown("#### Distribution des classes prédites")
    if dist:
        labels = list(dist.keys())
        values = list(dist.values())
        colors = {
            "normal": "#38a169",
            "suspected_opacity": "#e53e3e",
            "uncertain": "#dd6b20",
        }
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            values,
            labels=labels,
            colors=[colors.get(l, "#718096") for l in labels],
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title("Répartition", fontsize=12)
        st.pyplot(fig, use_container_width=True)

with col_table:
    st.markdown("#### Dernières analyses")
    runs = get_recent_runs(limit=50)
    if runs:
        df = pd.DataFrame(runs)
        df = df.rename(columns={
            "id": "ID",
            "case_id": "Image",
            "model_name": "Modèle",
            "prompt_version": "Version",
            "predicted_class": "Classe",
            "confidence": "Confiance",
            "latency_ms": "Latence (ms)",
            "created_at": "Date/heure",
        })
        df["Confiance"] = (df["Confiance"] * 100).round(1).astype(str) + " %"

        def color_class(val):
            c = {"Normal": "color: #38a169", "suspected_opacity": "color: #e53e3e",
                 "uncertain": "color: #dd6b20", "normal": "color: #38a169"}
            return c.get(val, "")

        st.dataframe(
            df[["ID", "Image", "Classe", "Confiance", "Latence (ms)", "Version", "Date/heure"]],
            use_container_width=True,
            height=350,
        )

st.markdown("---")

# --- Détail d'une ligne ---
st.markdown("#### Détail d'une prédiction")
runs_all = get_recent_runs(limit=200)
if runs_all:
    ids = [r["id"] for r in runs_all]
    selected_id = st.selectbox("Sélectionner un ID", ids)
    selected = next((r for r in runs_all if r["id"] == selected_id), None)
    if selected:
        st.json(selected)
