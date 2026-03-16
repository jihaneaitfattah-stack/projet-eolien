import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
from windrose import WindroseAxes

st.set_page_config(page_title="Analyse du potentiel éolien", layout="wide")
st.title("🌬 Analyse du potentiel éolien")

# =======================
# Instructions pour les étudiants
# =======================
st.markdown("""
Bienvenue dans l'application d'analyse du potentiel éolien !  
**Instructions :**  
1. Utilisez la **sidebar** pour saisir la latitude, la longitude et la période d'analyse.  
2. Sélectionnez les turbines à comparer.  
3. Cliquez sur **Lancer l'analyse** pour obtenir les résultats.  
4. Les graphiques et tableaux sont disponibles dans les onglets.  
5. Vous pouvez télécharger le tableau récapitulatif des turbines.
""")

# =======================
# Sidebar pour les paramètres
# =======================
st.sidebar.header("Paramètres du site")

# Utiliser session_state pour le bouton exemple
if 'lat' not in st.session_state:
    st.session_state['lat'] = 23.69
if 'lon' not in st.session_state:
    st.session_state['lon'] = -15.95
if 'start' not in st.session_state:
    st.session_state['start'] = "20230101"
if 'end' not in st.session_state:
    st.session_state['end'] = "20230201"

if st.sidebar.button("Exemple de site(beni mellal)"):
    st.session_state['lat'] = 32.3375
    st.session_state['lon'] = -6.3498
    st.session_state['start'] = "20230101"
    st.session_state['end'] = "20230201"

lat = st.sidebar.number_input("Latitude", value=st.session_state['lat'])
lon = st.sidebar.number_input("Longitude", value=st.session_state['lon'])
start = st.sidebar.text_input("Date début (YYYYMMDD)", st.session_state['start'])
end = st.sidebar.text_input("Date fin (YYYYMMDD)", st.session_state['end'])

# Sélection des turbines
turbines = {
    "Vestas_V52":{"P_nom":850000,"cut_in":3,"R":26},
    "Enercon_E40":{"P_nom":500000,"cut_in":2.5,"R":20},
    "Nordex_N50":{"P_nom":800000,"cut_in":3,"R":25},
    "GE_1_5s":{"P_nom":1500000,"cut_in":3.5,"R":38.5}
}

turbine_choice = st.sidebar.multiselect(
    "Sélectionnez les turbines à comparer",
    options=list(turbines.keys()),
    default=list(turbines.keys())
)

if st.sidebar.button("Lancer l'analyse"):

    # =======================
    # Récupération des données
    # =======================
    url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=WS50M,WS10M,WD50M&community=RE&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON"
    response = requests.get(url)
    data = response.json()

    ws50 = data["properties"]["parameter"]["WS50M"]
    ws10 = data["properties"]["parameter"]["WS10M"]
    wd50 = data["properties"]["parameter"]["WD50M"]

    df = pd.DataFrame({
        "wind5": ws50,
        "wind1": ws10,
        "direction": wd50
    })
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")

    # =======================
    # Statistiques
    # =======================
    v_mean5 = df["wind5"].mean()
    v_mean2 = df["wind1"].mean()
    v_std = df["wind5"].std()
    alpha = math.log(v_mean5/v_mean2)/math.log(50/10)

    terrains = [
        ("Lac, océan",0.10,0.0002),
        ("Herbes",0.15,0.03),
        ("Arbustes",0.20,0.10),
        ("Forêts",0.25,0.40),
        ("Petites villes",0.30,0.80),
        ("Grandes villes",0.40,1.80)
    ]

    terrain_proche = min(terrains, key=lambda x: abs(x[1]-alpha))
    type_terrain, alpha_ref, z0 = terrain_proche

    terrain_info = {
        "Lac, océan": "💧",
        "Herbes": "🌿",
        "Arbustes": "🌳",
        "Forêts": "🌲",
        "Petites villes": "🏘️",
        "Grandes villes": "🏙️"
    }

    # =======================
    # Affichage des statistiques
    # =======================
    st.subheader("Résultats statistiques")
    col1, col2, col3 = st.columns(3)
    col1.metric("Vitesse moyenne 50m", f"{v_mean5:.2f} m/s")
    col2.metric("Vitesse moyenne 10m", f"{v_mean2:.2f} m/s")
    col3.metric("Écart type", f"{v_std:.2f}")
    st.write("Coefficient alpha :", round(alpha,2))
    st.info(f"Type de terrain : {type_terrain} {terrain_info[type_terrain]}")

    # =======================
    # Paramètres Weibull
    # =======================
    params = weibull_min.fit(df["wind5"], floc=0)
    k, loc, c = params
    st.write("Paramètre k :", round(k,2))
    st.write("Paramètre c :", round(c,2))

    # =======================
    # Graphiques dans onglets
    # =======================
    tab1, tab2, tab3, tab4 = st.tabs(["Weibull", "Windrose", "Turbines", "Vitesse du vent"])

    # ---- Graphique Weibull ----
    v = np.linspace(0, df["wind5"].max(), 1000)
    f = (k/c)*(v/c)**(k-1)*np.exp(-(v/c)**k)
    fig, ax = plt.subplots(figsize=(4,2))
    ax.hist(df["wind5"], bins=30, density=True, alpha=0.6, color='skyblue')
    ax.plot(v, f, 'r', lw=2)
    ax.set_title("Distribution de Weibull")
    ax.set_xlabel("Vitesse (m/s)")
    ax.set_ylabel("Densité")
    with tab1:
        st.pyplot(fig)

    # ---- Windrose ----
    fig2 = plt.figure(figsize=(4,2))
    ax2 = WindroseAxes.from_ax(fig=fig2)
    ax2.bar(df["direction"], df["wind5"], normed=True, opening=0.8, edgecolor='white')
    ax2.set_legend()
    with tab2:
        st.pyplot(fig2)

    # ---- Comparaison des turbines ----
    rho = 1.225
    summary = []

    for name in turbine_choice:
        t = turbines[name]
        R = t["R"]
        A = np.pi * R**2

        P_site = 0.5 * rho * A * (df["wind5"]**3)
        P_mean = P_site.mean()
        TPI = P_mean / t["P_nom"]

        P_recup = np.minimum(P_site, t["P_nom"])
        P_moy_recup = P_recup.mean()
        E_total = P_recup.sum() / 1000

        summary.append({
            "Turbine": name,
            "P moy (W)": round(P_mean,0),
            "P moy récup (W)": round(P_moy_recup,0),
            "TPI": round(TPI,2),
            "Energie (kWh)": round(E_total,0)
        })

    df_summary = pd.DataFrame(summary)

    with tab3:
        st.dataframe(df_summary)
        # Graphique turbines
        fig3, ax3 = plt.subplots(figsize=(4,2))
        ax3.bar(df_summary["Turbine"], df_summary["P moy récup (W)"]/1000, color='green', alpha=0.7)
        ax3.set_ylabel("Puissance moyenne récupérable (kW)")
        ax3.set_title("Comparaison des turbines")
        st.pyplot(fig3)

        # Meilleure turbine
        best_turbine = df_summary.loc[df_summary["TPI"].idxmax(), "Turbine"]
        st.success(f"Meilleure turbine pour ce site : {best_turbine}")

        # Télécharger les données
        csv = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger le résumé des turbines",
            data=csv,
            file_name='resume_turbines.csv',
            mime='text/csv',
        )

    # ---- Graphique vitesse du vent ----
    fig4, ax4 = plt.subplots(figsize=(5,3))
    ax4.plot(df.index, df["wind5"], label="50m", color='blue')
    ax4.plot(df.index, df["wind1"], label="10m", color='orange')
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Vitesse du vent (m/s)")
    ax4.set_title("Vitesse du vent au fil du temps")
    ax4.legend()
    with tab4:
        st.pyplot(fig4)
