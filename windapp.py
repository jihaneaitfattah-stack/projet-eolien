import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
from windrose import WindroseAxes

st.title("🌬 Analyse du potentiel éolien")

# Sidebar pour les paramètres
st.sidebar.header("Paramètres du site")

lat = st.sidebar.number_input("Latitude", value=23.69)
lon = st.sidebar.number_input("Longitude", value=-15.95)

start = st.sidebar.text_input("Date début (YYYYMMDD)", "20230101")
end = st.sidebar.text_input("Date fin (YYYYMMDD)", "20230201")

if st.sidebar.button("Lancer l'analyse"):

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

    # statistiques
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

    st.subheader("Résultats statistiques")

    col1,col2,col3 = st.columns(3)

    col1.metric("Vitesse moyenne 50m",f"{v_mean5:.2f} m/s")
    col2.metric("Vitesse moyenne 10m",f"{v_mean2:.2f} m/s")
    col3.metric("Ecart type",f"{v_std:.2f}")

    st.write("Coefficient alpha :",round(alpha,2))
    st.write("Type de terrain :",type_terrain)

    # Weibull
    params = weibull_min.fit(df["wind5"], floc=0)
    k, loc, c = params

    st.write("Paramètre k :",round(k,2))
    st.write("Paramètre c :",round(c,2))

    # Graphique Weibull
    v = np.linspace(0,df["wind5"].max(),1000)
    f=(k/c)*(v/c)**(k-1)*np.exp(-(v/c)**k)

    fig,ax=plt.subplots()

    ax.hist(df["wind5"],bins=30,density=True)
    ax.plot(v,f)
    ax.set_title("Distribution de Weibull")
    ax.set_xlabel("Vitesse (m/s)")
    ax.set_ylabel("Densité")

    st.pyplot(fig)

    # Windrose
    fig2 = plt.figure()
    ax2 = WindroseAxes.from_ax()

    ax2.bar(df["direction"], df["wind5"],
            normed=True,
            opening=0.8,
            edgecolor='white')

    ax2.set_legend()

    st.pyplot(fig2)

    # Turbines
    turbines = {

    "Vestas_V52":{"P_nom":850000,"cut_in":3,"R":26},
    "Enercon_E40":{"P_nom":500000,"cut_in":2.5,"R":20},
    "Nordex_N50":{"P_nom":800000,"cut_in":3,"R":25},
    "GE_1_5s":{"P_nom":1500000,"cut_in":3.5,"R":38.5}

    }

    rho=1.225
    summary=[]

    for name,t in turbines.items():

        R=t["R"]
        A=np.pi*R**2

        P_site=0.5*rho*A*(df["wind5"]**3)
        P_mean=P_site.mean()

        TPI=P_mean/t["P_nom"]

        P_recup=np.minimum(P_site,t["P_nom"])
        P_moy_recup=P_recup.mean()

        E_total=P_recup.sum()/1000

        summary.append({
        "Turbine":name,
        "P moy (W)":round(P_mean,0),
        "P moy récup (W)":round(P_moy_recup,0),
        "TPI":round(TPI,2),
        "Energie (kWh)":round(E_total,0)
        })

    df_summary=pd.DataFrame(summary)

    st.subheader("Comparaison des turbines")

    st.dataframe(df_summary)

    # Graphique turbines
    fig3,ax3=plt.subplots()

    ax3.bar(df_summary["Turbine"],
            df_summary["P moy récup (W)"]/1000)

    ax3.set_ylabel("Puissance moyenne récupérable (kW)")
    ax3.set_title("Comparaison des turbines")

    st.pyplot(fig3)

    best_turbine = df_summary.loc[df_summary["TPI"].idxmax(),"Turbine"]

    st.success(f"Meilleure turbine pour ce site : {best_turbine}")
