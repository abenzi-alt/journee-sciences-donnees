import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
st.set_page_config(page_title="Sciences des données en action",layout="wide")
st.title("Les sciences des données en action")
st.write("Quatre mini-expériences interactives : prévoir, recommander, reconnaître et explorer.")
page=st.sidebar.radio("Choisir une expérience",["Prévoir les ventes","Recommander des livres","Reconnaître une image","Explorer les feux de forêt"])

if page=="Prévoir les ventes":
    st.header("Prévoir les ventes d'un magasin")
    st.write("Données synthétiques pour illustrer un modèle de régression.")
    rng=np.random.default_rng(7); n=300; jour=np.arange(n)
    df=pd.DataFrame({"jour":jour,"promotion":rng.choice([0,5,10,15,20,25,30],n),"weekend":(jour%7>=5).astype(int)})
    df["saison"]=np.sin(2*np.pi*jour/365)
    df["ventes"]=np.maximum(5,70+1.15*df.promotion+18*df.weekend+12*df.saison+rng.normal(0,8,n)).round()
    split=240; X=df[["promotion","weekend","saison"]]; y=df.ventes
    model=RandomForestRegressor(n_estimators=100,random_state=42,min_samples_leaf=3).fit(X.iloc[:split],y.iloc[:split])
    mae=mean_absolute_error(y.iloc[split:],model.predict(X.iloc[split:]))
    a,b=st.columns(2); a.metric("Jours simulés",len(df)); b.metric("Erreur moyenne sur la période test",f"{mae:.1f} unités")
    st.plotly_chart(px.line(df,x="jour",y="ventes",title="Historique simulé des ventes"),use_container_width=True)
    promo=st.slider("Promotion (%)",0,30,10,5); weekend=st.selectbox("Jour",["Semaine","Fin de semaine"]); saison=st.selectbox("Saison",["Hiver","Printemps","Été","Automne"])
    sm={"Hiver":-.7,"Printemps":.7,"Été":.9,"Automne":-.2}
    pred=model.predict(pd.DataFrame({"promotion":[promo],"weekend":[int(weekend=="Fin de semaine")],"saison":[sm[saison]]}))[0]
    st.success(f"Prévision illustrative : environ {pred:.0f} unités.")
    st.caption("Jeu synthétique : cette prévision n'est pas destinée à une décision commerciale réelle.")

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
    st.header("Explorer les données des feux de forêt")
    st.write("Comparez des distributions météorologiques entre classes. Vous pouvez importer votre fichier réel.")
    file=st.file_uploader("Importer algerian_data.csv (facultatif)",type=["csv"])
    if file:
        df=pd.read_csv(file)
    else:
        rng=np.random.default_rng(22); n=240
        classes=rng.choice(["Feu","Non feu"],n,p=[.55,.45])
        df=pd.DataFrame({"Temp":np.where(classes=="Feu",rng.normal(33,4,n),rng.normal(29,5,n)).clip(20,45),
                         "RH":np.where(classes=="Feu",rng.normal(40,12,n),rng.normal(58,15,n)).clip(15,95),
                         "Ws":rng.normal(15,4,n).clip(4,35),"Classes":classes})
        st.caption("Jeu synthétique : tendances inventées pour montrer l'interface, pas des conclusions scientifiques.")
    st.write(f"{len(df)} lignes · {len(df.columns)} colonnes")
    st.dataframe(df.head(),use_container_width=True)
    nums=df.select_dtypes(include=np.number).columns.tolist()
    cats=df.select_dtypes(exclude=np.number).columns.tolist()
    if nums:
        a,b=st.columns(2)
        var=a.selectbox("Variable numérique",nums)
        grp=b.selectbox("Comparer selon",["Aucune"]+cats)
        if grp!="Aucune": fig=px.box(df.dropna(subset=[var,grp]),x=grp,y=var,color=grp,title=f"{var} selon {grp}",points="outliers")
        else: fig=px.histogram(df,x=var,marginal="box",title=f"Distribution de {var}")
        st.plotly_chart(fig,use_container_width=True)
        a,b,c=st.columns(3); a.metric("Médiane",f"{df[var].median():.2f}"); b.metric("Valeurs manquantes",int(df[var].isna().sum())); c.metric("Valeurs distinctes",int(df[var].nunique()))
    if cats:
        cat=st.selectbox("Variable qualitative à compter",cats)
        st.plotly_chart(px.histogram(df,x=cat,color=cat,title=f"Répartition de {cat}"),use_container_width=True)
    st.download_button("Télécharger les données",df.to_csv(index=False).encode("utf-8"),"donnees.csv","text/csv")
    st.caption("Une association observée ne prouve pas un lien de causalité.")
