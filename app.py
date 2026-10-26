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
    st.header("Tableau de bord — Journée des sciences des données")
    st.write(
        "Vue d'ensemble des quatre expériences interactives "
        "et de leurs objectifs."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Expériences disponibles", "4")
    c2.metric("Domaines illustrés", "3")
    c3.metric("Objectif", "Découvrir")

    st.subheader("Les expériences proposées")

    experiences = pd.DataFrame({
        "Expérience": [
            "Prévoir les ventes",
            "Recommander des livres",
            "Reconnaître une image",
            "Tableau de bord"
        ],
        "Domaine": [
            "Prévision et régression",
            "Systèmes de recommandation",
            "Classification d'images",
            "Synthèse et visualisation"
        ],
        "Objectif": [
            "Comparer les prévisions et les scénarios",
            "Explorer des recommandations selon ses goûts",
            "Comparer les catégories proposées par un modèle",
            "Consulter la synthèse des activités"
        ]
    })

    st.dataframe(experiences, hide_index=True, use_container_width=True)

    st.subheader("Répartition des expériences")
    st.plotly_chart(
        px.bar(
            experiences,
            x="Expérience",
            y=[1, 1, 1, 1],
            title="Une interface par expérience"
        ),
        use_container_width=True
    )

    st.info(
        "Utilisez le menu à gauche pour ouvrir une expérience et "
        "tester ses paramètres."
    )

