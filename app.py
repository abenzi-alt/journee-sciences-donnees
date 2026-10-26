import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
st.set_page_config(page_title="Sciences des données en action",layout="wide")
st.title("Les sciences des données en action")
st.write("Quatre mini-expériences interactives : prévoir, recommander, reconnaître et explorer.")
page=st.sidebar.radio("Choisir une expérience",["Prévoir les ventes","Recommander des livres","Reconnaître une image","Tableau de bord"])

# Configuration visuelle du tableau de bord
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f7ff 0%, #fff8e7 100%);
    }

    h1 {
        color: #5425D8 !important;
        font-weight: 900 !important;
    }

    h2, h3 {
        color: #0756C9 !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #5425D8, #087CFA);
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 5px 14px #5425D833;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #FFE45E !important;
        font-weight: 900 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #B8FFD8 !important;
    }

    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, #FF6B00, #FF2D75);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 800;
    }

    div[data-testid="stDownloadButton"] button:hover {
        background: #5425D8;
        color: white;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


if page == "Prévoir les ventes":
    st.header("Prévoir les ventes d'un magasin")
    st.write("Modifiez les paramètres et observez leurs effets sur les prévisions.")

    with st.sidebar:
        st.subheader("Paramètres de simulation")
        n = st.selectbox("Historique (jours)", [300, 730, 1095], index=1)
        promo = st.slider("Promotion (%)", 0, 50, 10, 5)
        jour_type = st.selectbox(
            "Type de jour",
            ["Jour de semaine", "Fin de semaine", "Jour férié"],
            index=1
        )
        saison = st.selectbox(
            "Saison", ["Hiver", "Printemps", "Été", "Automne"], index=2
        )
        methode = st.selectbox(
            "Méthode",
            ["Régression linéaire", "Forêt aléatoire"]
        )
        horizon = st.selectbox("Horizon de prévision", [30, 60, 90, 180], index=2)

    # Génération d'un historique synthétique reproductible
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"jour": np.arange(n)})
    df["promotion"] = rng.choice([0, 5, 10, 15, 20, 25, 30], n)
    df["weekend"] = (df["jour"] % 7 >= 5).astype(int)
    df["saison_num"] = np.sin(2 * np.pi * df["jour"] / 365)
    df["ventes"] = np.maximum(
        5,
        70 + 0.8 * df["promotion"]
        + 18 * df["weekend"]
        + 12 * df["saison_num"]
        + 0.025 * df["jour"]
        + rng.normal(0, 8, n)
    )

    variables = ["jour", "promotion", "weekend", "saison_num"]
    split = int(n * 0.8)
    X_train, X_test = df[variables].iloc[:split], df[variables].iloc[split:]
    y_train, y_test = df["ventes"].iloc[:split], df["ventes"].iloc[split:]

    if methode == "Régression linéaire":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=150, random_state=42, min_samples_leaf=3
        )

    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))

    c1, c2, c3 = st.columns(3)
    c1.metric("Jours d'historique", n)
    c2.metric("MAE sur le test", f"{mae:.1f} unités")
    c3.metric("RMSE sur le test", f"{rmse:.1f} unités")

    st.subheader("1. Ventes observées et prévisions sur le test")
    test_graph = pd.DataFrame({
        "Jour": X_test["jour"].values,
        "Ventes observées": y_test.values,
        "Ventes prévues": pred_test
    }).melt(
        id_vars="Jour", var_name="Série", value_name="Ventes"
    )
    st.plotly_chart(
        px.line(test_graph, x="Jour", y="Ventes", color="Série"),
        use_container_width=True
    )

    # Simulation prospective
    jours_futurs = np.arange(n, n + horizon)
    saison_val = {
        "Hiver": -0.7, "Printemps": 0.7,
        "Été": 0.9, "Automne": -0.2
    }
    weekend_val = {
        "Jour de semaine": 0,
        "Fin de semaine": 1,
        "Jour férié": 1
    }

    scenario = pd.DataFrame({
        "jour": jours_futurs,
        "promotion": promo,
        "weekend": weekend_val[jour_type],
        "saison_num": saison_val[saison]
    })
    scenario["Ventes prévues"] = np.maximum(
        0, model.predict(scenario[variables])
    )

    # Référence sans promotion, avec calendrier habituel
    reference = scenario[variables].copy()
    reference["promotion"] = 0
    reference["weekend"] = (jours_futurs % 7 >= 5).astype(int)
    reference["saison_num"] = np.sin(2 * np.pi * jours_futurs / 365)
    scenario["Référence"] = np.maximum(
        0, model.predict(reference[variables])
    )

    st.subheader("2. Comparer le scénario et la référence")
    comparaison = scenario.melt(
        id_vars="jour",
        value_vars=["Ventes prévues", "Référence"],
        var_name="Scénario",
        value_name="Ventes"
    )
    st.plotly_chart(
        px.line(
            comparaison, x="jour", y="Ventes", color="Scénario",
            title=f"Prévision sur {horizon} jours"
        ),
        use_container_width=True
    )

    a, b, c = st.columns(3)
    a.metric("Moyenne prévue", f"{scenario['Ventes prévues'].mean():.1f}")
    b.metric("Moyenne de référence", f"{scenario['Référence'].mean():.1f}")
    c.metric(
        "Écart moyen",
        f"{(scenario['Ventes prévues'] - scenario['Référence']).mean():+.1f}"
    )

    st.download_button(
        "Télécharger les prévisions CSV",
        scenario.to_csv(index=False).encode("utf-8"),
        "previsions_ventes.csv",
        "text/csv"
    )
    st.caption(
        "Données synthétiques à but pédagogique. Le type de jour et la saison "
        "choisis sont appliqués à l'ensemble du scénario. Les écarts simulés "
        "ne prouvent pas un effet causal de la promotion."
    )


elif page=="Recommander des livres":
    st.header("Trouver son prochain livre")
    books=pd.DataFrame([
    ["Les mystères du phare","Policier","Enquête, indices, village côtier et suspense."],
    ["Le dernier témoin","Policier","Détective, affaire criminelle et enquête pleine de suspense."],
    ["L'ombre des archives","Policier","Une archiviste découvre des preuves oubliées."],
    ["Voyage vers Mars","Science-fiction","Mission spatiale, Mars, robots et vie dans l'univers."],
    ["Le code des étoiles","Science-fiction","Intelligence artificielle, exploration spatiale et avenir."],
    ["Les jardins de demain","Environnement","Biodiversité, agriculture urbaine et climat."],
    ["Comprendre le climat","Sciences","Données environnementales et changements climatiques."],
    ["Petite histoire des nombres","Mathématiques","Nombres, statistiques et usages des mathématiques."],
    ["Les données au quotidien","Sciences des données","Statistiques, visualisation et apprentissage automatique."],
    ["Le café des souvenirs","Roman","Amitié, rencontres et souvenirs."]
    ],columns=["Titre","Genre","Description"])
    books["texte"]=books.Genre+" "+books.Description
    mat=TfidfVectorizer().fit_transform(books.texte)
    selected=st.multiselect("Choisissez les livres que vous aimez",books.Titre.tolist(),default=[books.Titre.iloc[3]])
    genre=st.selectbox("Genre souhaité",["Tous"]+sorted(books.Genre.unique()))
    if st.button("Obtenir des recommandations",type="primary"):
        if not selected: st.warning("Choisissez au moins un livre.")
        else:
            ids=books.index[books.Titre.isin(selected)].tolist()
            profile=np.asarray(mat[ids].mean(axis=0))
            scores=cosine_similarity(profile,mat).ravel()
            rec=books.copy(); rec["Similarité"]=scores
            rec=rec[~rec.Titre.isin(selected)]
            if genre!="Tous": rec=rec[rec.Genre==genre]
            rec=rec.sort_values("Similarité",ascending=False).head(5)
            if rec.empty: st.info("Aucune suggestion pour ce filtre.")
            for _,r in rec.iterrows():
                with st.container(border=True):
                    st.subheader(r.Titre); st.write(r.Description); st.caption(f"{r.Genre} · proximité textuelle : {r.Similarité:.0%}")
    st.caption("Petit catalogue pédagogique. La similarité textuelle ne garantit pas que vous aimerez le livre.")

elif page=="Reconnaître une image":
    st.header("Faire reconnaître une image par l'IA")
    st.write("Cette page utilise MobileNetV2 préentraîné. Le premier chargement peut être long et nécessite Internet.")
    file=st.file_uploader("Choisir une image JPG ou PNG",type=["jpg","jpeg","png"])
    if file:
        from PIL import Image
        img=Image.open(file).convert("RGB"); st.image(img,caption="Image choisie",width=450)
        if st.button("Analyser l'image",type="primary"):
            try:
                import torch
                from torchvision.models import mobilenet_v2,MobileNet_V2_Weights
                weights=MobileNet_V2_Weights.DEFAULT; model=mobilenet_v2(weights=weights).eval()
                tensor=weights.transforms()(img).unsqueeze(0)
                with torch.no_grad(): probs=model(tensor).softmax(1)[0]
                vals,inds=probs.topk(5); labels=weights.meta["categories"]
                result=pd.DataFrame({"Catégorie":[labels[i] for i in inds.tolist()],"Score":[float(v) for v in vals]})
                st.dataframe(result.style.format({"Score":"{:.1%}"}),hide_index=True,use_container_width=True)
                st.bar_chart(result.set_index("Catégorie"))
            except Exception as e:
                st.error("Le modèle n'a pas pu être chargé. Testez l'installation de PyTorch/torchvision avant la journée.")
                st.code(str(e))
    st.caption("Modèle généraliste : les scores ne garantissent pas la justesse. Ne pas utiliser pour un diagnostic ou une décision à risque.")
    

else:
    st.title("Météo et climat : évolution au fil des années")
    st.write(
        "Explorez les variations de l'ensoleillement, des précipitations, "
        "des températures et des épisodes météorologiques intenses."
    )

    # Villes proposées : coordonnées géographiques
    villes = {
        "Ottawa": (45.4215, -75.6972),
        "Montréal": (45.5019, -73.5674),
        "Toronto": (43.6532, -79.3832),
        "Québec": (46.8139, -71.2080),
        "Vancouver": (49.2827, -123.1207),
        "Paris": (48.8566, 2.3522),
    }

    c1, c2 = st.columns(2)

    with c1:
        ville = st.selectbox("Choisir une ville", list(villes.keys()))

    with c2:
        periode = st.slider(
            "Période étudiée",
            min_value=2000,
            max_value=2025,
            value=(2010, 2025)
        )

    latitude, longitude = villes[ville]
    debut, fin = periode

    @st.cache_data(ttl=86400)
    def charger_meteo(lat, lon, annee_debut, annee_fin):
        import requests

        url = "https://archive-api.open-meteo.com/v1/archive"

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{annee_debut}-01-01",
            "end_date": f"{annee_fin}-12-31",
            "daily": (
                "weather_code,"
                "temperature_2m_mean,"
                "temperature_2m_max,"
                "precipitation_sum,"
                "rain_sum,"
                "snowfall_sum,"
                "precipitation_hours,"
                "sunshine_duration,"
                "wind_gusts_10m_max"
            ),
            "timezone": "auto"
        }

        reponse = requests.get(url, params=params, timeout=60)
        reponse.raise_for_status()
        donnees = reponse.json()

        if "daily" not in donnees:
            raise ValueError("Les données météorologiques sont indisponibles.")

        df = pd.DataFrame(donnees["daily"])
        df["time"] = pd.to_datetime(df["time"])
        return df

    try:
        with st.spinner("Chargement des données météorologiques..."):
            df = charger_meteo(
                latitude, longitude, debut, fin
            )

        if df.empty:
            st.warning("Aucune donnée disponible pour cette période.")
            st.stop()

        # Préparation des variables
        df["annee"] = df["time"].dt.year
        df["ensoleillement_h"] = df["sunshine_duration"] / 3600

        # Définitions opérationnelles des indicateurs
        df["jour_ensoleille"] = (
            df["ensoleillement_h"] >= 6
        ).astype(int)

        df["jour_pluvieux"] = (
            df["precipitation_sum"] > 1
        ).astype(int)

        # Orage selon le code météo, ou rafales >= 70 km/h
        df["jour_intense"] = (
            df["weather_code"].isin([95, 96, 99])
            | (df["wind_gusts_10m_max"] >= 70)
        ).astype(int)

        # Agrégation annuelle
        annuel = (
            df.groupby("annee")
            .agg(
                Jours_ensoleilles=("jour_ensoleille", "sum"),
                Ensoleillement_h=("ensoleillement_h", "sum"),
                Precipitations_mm=("precipitation_sum", "sum"),
                Jours_pluvieux=("jour_pluvieux", "sum"),
                Heures_precipitations=("precipitation_hours", "sum"),
                Jours_orage_ou_vents_forts=("jour_intense", "sum"),
                Temperature_moyenne_C=("temperature_2m_mean", "mean"),
                Temperature_maximale_C=("temperature_2m_max", "max"),
                Neige_cm=("snowfall_sum", "sum"),
                Rafale_max_kmh=("wind_gusts_10m_max", "max")
            )
            .reset_index()
        )

        st.subheader(f"Vue d'ensemble — {ville}")

        total_soleil = int(annuel["Jours_ensoleilles"].sum())
        total_pluie = round(annuel["Precipitations_mm"].sum(), 1)
        total_intense = int(
            annuel["Jours_orage_ou_vents_forts"].sum()
        )
        temp_moyenne = round(
            df["temperature_2m_mean"].mean(), 1
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jours ensoleillés", f"{total_soleil:,}")
        m2.metric("Précipitations cumulées", f"{total_pluie:,.0f} mm")
        m3.metric("Jours météo intense", f"{total_intense:,}")
        m4.metric("Température moyenne", f"{temp_moyenne} °C")

        st.caption(
            "Les indicateurs couvrent la période sélectionnée. "
            "Un jour ensoleillé correspond à au moins 6 h de soleil. "
            "Un jour météo intense correspond à un code d'orage "
            "(95, 96 ou 99) ou à des rafales d'au moins 70 km/h. "
            "Ces indicateurs ne représentent pas un inventaire exhaustif "
            "de toutes les tempêtes."
        )

        st.divider()
        st.subheader("1. Combien de jours ensoleillés par année ?")

        fig_soleil = px.bar(
            annuel,
            x="annee",
            y="Jours_ensoleilles",
            color_discrete_sequence=["#FFD600"],
            labels={
                "annee": "Année",
                "Jours_ensoleilles": "Nombre de jours"
            },
            title="Évolution annuelle des jours ensoleillés"
        )
        st.plotly_chart(fig_soleil, use_container_width=True)

        st.subheader("2. Les précipitations augmentent-elles ?")

        fig_pluie = px.bar(
            annuel,
            x="annee",
            y="Precipitations_mm",
            labels={
                "annee": "Année",
                "Precipitations_mm": "Précipitations (mm)"
            },
            title="Cumul annuel des précipitations"
        )
        st.plotly_chart(fig_pluie, use_container_width=True)

        st.subheader("3. Comment évoluent les températures ?")

        fig_temp = px.line(
            annuel,
            x="annee",
            y=["Temperature_moyenne_C", "Temperature_maximale_C"],
            markers=True,
            labels={
                "annee": "Année",
                "value": "Température (°C)",
                "variable": "Indicateur"
            },
            title="Températures annuelles"
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        st.subheader("4. Combien de jours avec orages ou vents forts ?")

        fig_intense = px.bar(
            annuel,
            x="annee",
            y="Jours_orage_ou_vents_forts",
            labels={
                "annee": "Année",
                "Jours_orage_ou_vents_forts": "Nombre de jours"
            },
            title="Jours d'orage ou de rafales fortes"
        )
        st.plotly_chart(fig_intense, use_container_width=True)

        st.subheader("5. Tableau des indicateurs annuels")

        st.dataframe(
            annuel.round(1),
            hide_index=True,
            use_container_width=True
        )

        fichier_csv = annuel.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "Télécharger les données annuelles (CSV)",
            data=fichier_csv,
            file_name=f"meteo_{ville}_{debut}_{fin}.csv",
            mime="text/csv"
        )

    except Exception as erreur:
        st.error(
            "Impossible de charger les données météo. "
            "Vérifiez la connexion Internet et réessayez."
        )
        st.caption(f"Détail technique : {erreur}")
