"""
Application Streamlit — Détection de Transactions Bancaires Frauduleuses
Master MSDE 7 — EHTP — Module 5 Machine Learning
Encadrant : Pr. Abdelhamid Fadil
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# CSS personnalisé pour l'esthétique
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DU PIPELINE (avec cache pour éviter de recharger)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load('fraud_detection_pipeline.pkl')

try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Erreur de chargement du modèle : {e}")
    model_loaded = False

# ─────────────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">💳 Détection de Transactions Bancaires Frauduleuses</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Application de prédiction en temps réel basée sur un modèle Extra Trees tuné</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# INFORMATIONS SUR LE MODÈLE (panneau supérieur)
# ─────────────────────────────────────────────────────────────────
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("🎯 PR-AUC", "0.8806")
with col_m2:
    st.metric("📍 Recall", "84.69%")
with col_m3:
    st.metric("✅ Precision", "90.22%")
with col_m4:
    st.metric("⚡ F1-Score", "0.8737")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — SAISIE DES DONNÉES
# ─────────────────────────────────────────────────────────────────
st.sidebar.header("📝 Saisie de la transaction")
st.sidebar.markdown("Renseignez les caractéristiques de la transaction à analyser.")

# Mode de saisie rapide
st.sidebar.markdown("**Mode rapide :**")
mode = st.sidebar.radio(
    "Choisir un exemple :",
    ["Saisie manuelle", "Exemple Légitime", "Exemple Fraude"],
    index=0
)

# Valeurs par défaut selon le mode
if mode == "Exemple Légitime":
    default_time, default_amount = 50000.0, 100.0
    default_v = [-0.5, 0.3, 0.8, 0.1, -0.2, 0.4, -0.1, 0.05, -0.3,
                 0.2, 0.5, -0.1, 0.3, 0.1, -0.2, 0.4, 0.6, -0.2,
                 0.3, -0.1, -0.05, 0.1, 0.0, 0.2, -0.1, 0.0, 0.05, -0.05]
elif mode == "Exemple Fraude":
    default_time, default_amount = 100000.0, 250.0
    # Valeurs typiques d'une fraude (V14, V12, V10, V17 très négatifs)
    default_v = [-2.3, 2.8, -3.5, 2.1, -1.8, -0.9, -3.2, 1.5, -2.1,
                 -5.5, 3.2, -6.1, 0.2, -7.3, 0.1, -3.0, -7.8, -2.5,
                 1.2, 0.3, 0.5, 0.1, -0.2, 0.0, 0.1, -0.05, 0.2, 0.0]
else:
    default_time, default_amount = 0.0, 0.0
    default_v = [0.0] * 28

# Saisie Time et Amount
st.sidebar.markdown("**📊 Variables principales :**")
time_val = st.sidebar.number_input(
    "⏱️ Time (secondes depuis la 1ère transaction)",
    min_value=0.0, max_value=172800.0,
    value=float(default_time), step=100.0,
    help="Temps écoulé en secondes depuis la première transaction du dataset."
)
amount_val = st.sidebar.number_input(
    "💰 Amount (montant en €)",
    min_value=0.0, max_value=30000.0,
    value=float(default_amount), step=10.0,
    help="Montant de la transaction en euros."
)

# Saisie V1 à V28 dans un expander pour ne pas surcharger
st.sidebar.markdown("**🔢 Composantes PCA (V1 à V28) :**")
st.sidebar.caption("Variables issues d'une transformation PCA. Valeurs typiques autour de 0.")

v_values = {}
with st.sidebar.expander("📂 Voir/Modifier V1 à V28", expanded=False):
    # Disposition en 2 colonnes pour gagner de la place
    col_v1, col_v2 = st.columns(2)
    for i in range(1, 29):
        target_col = col_v1 if i <= 14 else col_v2
        with target_col:
            v_values[f'V{i}'] = st.number_input(
                f"V{i}",
                min_value=-50.0, max_value=50.0,
                value=float(default_v[i-1]),
                step=0.1,
                key=f"v{i}",
                format="%.2f"
            )

# Bouton de prédiction
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 ANALYSER LA TRANSACTION", type="primary")

# ─────────────────────────────────────────────────────────────────
# CORPS PRINCIPAL
# ─────────────────────────────────────────────────────────────────

# Construction du DataFrame d'entrée (dans l'ordre exact attendu par le pipeline)
columns_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_dict = {'Time': time_val, **v_values, 'Amount': amount_val}
input_df = pd.DataFrame([input_dict])[columns_order]

# Deux colonnes : données saisies | résultat
col_left, col_right = st.columns([1, 1])

# ── COLONNE GAUCHE : Données saisies
with col_left:
    st.subheader("📋 Données saisies")
    st.caption("Récapitulatif des valeurs renseignées :")

    # Affichage en 2 sous-colonnes
    main_vars = pd.DataFrame({
        'Variable': ['Time', 'Amount'],
        'Valeur': [f"{time_val:.2f} s", f"{amount_val:.2f} €"]
    })
    st.dataframe(main_vars, hide_index=True, use_container_width=True)

    with st.expander("📊 Voir les composantes PCA (V1-V28)", expanded=False):
        v_df = pd.DataFrame({
            'Variable': list(v_values.keys()),
            'Valeur': [f"{v:.4f}" for v in v_values.values()]
        })
        st.dataframe(v_df, hide_index=True, height=350, use_container_width=True)

# ── COLONNE DROITE : Résultat de la prédiction
with col_right:
    st.subheader("🎯 Résultat de l'analyse")

    if not model_loaded:
        st.warning("⚠️ Le modèle n'est pas chargé. Vérifiez que `fraud_detection_pipeline.pkl` est présent.")
    elif predict_btn:
        # Prédiction
        with st.spinner("Analyse en cours..."):
            prediction = pipeline.predict(input_df)[0]
            proba_fraud = pipeline.predict_proba(input_df)[0, 1]
            proba_legit = 1 - proba_fraud
            confidence  = max(proba_fraud, proba_legit)

        # Affichage du résultat principal
        if prediction == 1:
            st.error("🚨 **TRANSACTION FRAUDULEUSE DÉTECTÉE**")
            st.markdown(f"""
            <div style='background-color:#ffe6e6; padding:1.5rem; border-radius:10px; border-left:5px solid #d32f2f;'>
                <h3 style='color:#d32f2f; margin-top:0;'>⚠️ Alerte de fraude</h3>
                <p style='font-size:1.1rem; margin-bottom:0;'>Cette transaction présente un risque très élevé de fraude. Action recommandée : <b>blocage et vérification manuelle</b>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ **TRANSACTION LÉGITIME**")
            st.markdown(f"""
            <div style='background-color:#e6f7e6; padding:1.5rem; border-radius:10px; border-left:5px solid #2e7d32;'>
                <h3 style='color:#2e7d32; margin-top:0;'>✓ Transaction validée</h3>
                <p style='font-size:1.1rem; margin-bottom:0;'>Cette transaction présente les caractéristiques d'une opération normale. Aucune action particulière requise.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Niveau de confiance")

        # Barres de probabilité
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("Probabilité Légitime", f"{proba_legit*100:.2f}%")
            st.progress(float(proba_legit))
        with col_p2:
            st.metric("Probabilité Fraude", f"{proba_fraud*100:.2f}%")
            st.progress(float(proba_fraud))

        st.markdown(f"**Confiance du modèle :** `{confidence*100:.1f}%`")

        # Détails techniques
        with st.expander("🔬 Détails techniques", expanded=False):
            st.write(f"- **Classe prédite :** `{int(prediction)}` ({'Fraude' if prediction == 1 else 'Légitime'})")
            st.write(f"- **P(Légitime)** : `{proba_legit:.6f}`")
            st.write(f"- **P(Fraude)**   : `{proba_fraud:.6f}`")
            st.write(f"- **Seuil de décision** : `0.50`")
            st.write(f"- **Modèle utilisé** : Extra Trees (300 arbres, tuné par RandomizedSearchCV)")
    else:
        st.info("👈 Renseignez les données dans la barre latérale, puis cliquez sur **ANALYSER LA TRANSACTION**.")
        st.markdown("""
        💡 **Astuce** : utilisez les boutons "Exemple Légitime" ou "Exemple Fraude" dans la sidebar pour tester rapidement.
        """)

# ─────────────────────────────────────────────────────────────────
# SECTION INFORMATIONS — En bas de page
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ À propos de ce modèle"):
    st.markdown("""
    ### 🎓 Contexte du projet

    Cette application est issue d'un projet académique réalisé dans le cadre du
    **Master en Data Engineering (MSDE 7)** à l'**École Hassania des Travaux Publics (EHTP)**.

    ### 📊 Dataset

    - **Source** : *Credit Card Fraud Detection* (Kaggle, septembre 2013)
    - **Taille** : 284 807 transactions
    - **Features** : 30 (Time, Amount, V1-V28 issues d'une PCA)
    - **Déséquilibre** : seulement **0.172%** de fraudes

    ### 🛠️ Méthodologie

    1. **Preprocessing** : RobustScaler sur Amount et Time, SMOTE pour rééquilibrer les classes
    2. **Comparaison de 11 algorithmes ML** (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, etc.)
    3. **Tuning** des 3 meilleurs modèles par RandomizedSearchCV
    4. **Modèle final retenu** : **Extra Trees** (300 arbres, max_depth illimité)

    ### 📈 Performances finales

    | Métrique | Valeur |
    |----------|--------|
    | PR-AUC | 0.8806 |
    | Recall | 84.69% |
    | Precision | 90.22% |
    | F1-Score | 0.8737 |
    | G-Mean | 0.9202 |

    ### 👥 Auteurs

    - **FANANE SARA**
    - **MOUNIR MARIA**

    """)

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85em; padding-top: 2rem;'>
    💳 Application de détection de fraude bancaire — MSDE 7 EHTP — 2025/2026
</div>
""", unsafe_allow_html=True)