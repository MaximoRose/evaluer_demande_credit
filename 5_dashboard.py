import streamlit as st
import pandas as pd
import mbr_kernel as mkn
import os.path
import time

CURRENT_DATA_PATH = './st_content/processed_new_data.csv'


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    CORTEX_URI = 'http://0.0.0.0:8890/'
    RAY_SERVE_URI = 'http://127.0.0.1:8000/regressor'

    new_data = None
    predict_btn = None

    # ----------- Sidebar
    page = st.sidebar.selectbox('Navigation', ["Evaluer un dossier", "Charger donnees", "Predire lot"])

    st.sidebar.markdown("""---""")
    st.sidebar.write("Créée by Massimo Bruni")
    # st.sidebar.image("assets/logo.png", width=100)
    if page == "Evaluer un dossier":
        st.title('Aide a la decision\nSolvabilité client·e')

        revenu_med = st.number_input('Revenu médian dans le secteur (en 10K de dollars)',
                                     min_value=0., value=3.87, step=1.)

        age_med = st.number_input('Âge médian des maisons dans le secteur',
                                  min_value=0., value=28., step=1.)

        nb_piece_med = st.number_input('Nombre moyen de pièces',
                                       min_value=0., value=5., step=1.)

        nb_chambre_moy = st.number_input('Nombre moyen de chambres',
                                         min_value=0., value=1., step=1.)

        taille_pop = st.number_input('Taille de la population dans le secteur',
                                     min_value=0, value=1425, step=100)

        occupation_moy = st.number_input('Occupation moyenne de la maison (en nombre d\'habitants)',
                                         min_value=0., value=3., step=1.)

        latitude = st.number_input('Latitude du secteur',
                                   value=35., step=1.)

        longitude = st.number_input('Longitude du secteur',
                                    value=-119., step=1.)

        predict_btn = st.button('Prédire')
    elif page == "Charger donnees":
        st.title('Charger un nouveau lot de dossiers clients')
        uploaded_file = st.file_uploader('Charger un fichier de donnees', type='csv')
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe)
            new_data = st.button('Charger les données')
            if new_data:
                with st.spinner("Operation en cours..."):
                    processed_new_data = mkn.full_feature_engineering(df_input=dataframe, df_folder='./input_data/',
                                                                      encoding_treshold=0.04, nan_treshold=0.4)
                st.success("Nouvelles donnees traitees pour prediction")
                processed_new_data.to_csv(CURRENT_DATA_PATH)
    elif page == "Predire lot":
        st.title('Generer les predictions pour un lot de dosssier')
        if os.path.exists(CURRENT_DATA_PATH):
            date_creation_fichier = time.ctime(os.path.getctime(CURRENT_DATA_PATH))
            st.write("Date de creation du dernier fichier : " + str(date_creation_fichier))
            predict_btn = st.button('Generer predictions')
            if predict_btn:
                with st.spinner("Operation en cours..."):
                    time.sleep(1)
                st.success("Un jeu de donnees predites a ete genere")
        else:
            st.markdown("Aucun jeu de donnees n'a encore ete charge. Utilisez la navigation pour charger un fichier")

    else:
        st.markdown("This page is not implemented yet :no_entry_sign:")


if __name__ == '__main__':
    main()
