import streamlit as st
import pandas as pd
import mbr_kernel as mkn
import os.path
import time
import json
import requests

PREPROCESSED_DATA_PATH = './st_content/processed_new_data.csv'
ORIGINAL_DATA_PATH = './st_content/original_data.csv'
CUSTOM_CSS = './st_content/style.css'
HEROKU_APP_PREDICTION_URI = "https://oc-ds-p7.herokuapp.com/solvability_prediction"


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    return


def predict_with_heroku_app(lst_columns, associated_data):
    dict_customer = {k: v for k, v in zip(lst_columns, associated_data)}
    input_json = json.dumps(dict_customer)
    response = requests.post(HEROKU_APP_PREDICTION_URI, data=input_json)
    return response.text


def main():
    local_css(CUSTOM_CSS)

    MODEL_URI = 'http://127.0.0.1:5001/invocations'

    new_data = None
    predict_btn = None
    sk_id_combo = None

    # ----------- Sidebar
    page = st.sidebar.selectbox('Navigation', ["Evaluer un dossier", "Charger donnees"])  # , "Predire lot"

    st.sidebar.markdown("""---""")
    st.sidebar.write("Créée by Massimo Bruni")
    # st.sidebar.image("assets/logo.png", width=100)
    if page == "Evaluer un dossier":
        st.title('Predire la solvabilité client·e')
        if os.path.exists(PREPROCESSED_DATA_PATH):
            date_creation_fichier = time.ctime(os.path.getctime(PREPROCESSED_DATA_PATH))
            st.write("Data's date : " + str(date_creation_fichier))
            current_data = pd.read_csv(PREPROCESSED_DATA_PATH)
            sk_id_combo = st.selectbox("Selectionnez l'identifiant client : ",
                                       current_data['SK_ID_CURR'].values.tolist())
            if sk_id_combo is not None:
                liste_colonnes = current_data.drop(columns='SK_ID_CURR').columns.tolist()
                data_client = current_data[current_data['SK_ID_CURR'] == sk_id_combo].drop(
                    columns=['SK_ID_CURR']).values

                prediction = predict_with_heroku_app(lst_columns=liste_colonnes,
                                                     associated_data=data_client.tolist()[0])

                if prediction == '0':
                    text_result = "<span class='ok_customer'> OK </span>"
                    st.markdown(text_result, unsafe_allow_html=True)
                else:
                    text_result = "<span class='nok_customer'> NOT OK </span>"
                    st.markdown(text_result, unsafe_allow_html=True)
        else:
            st.markdown("Aucun jeu de donnees n'a encore ete charge. Utilisez la navigation pour charger un fichier")

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
                processed_new_data.to_csv(PREPROCESSED_DATA_PATH, index=False)
                dataframe.to_csv(ORIGINAL_DATA_PATH, index=False)
    # elif page == "Predire lot":
    #     st.title('Generer les predictions pour un lot de dosssier')
    #     if os.path.exists(PREPROCESSED_DATA_PATH):
    #         date_creation_fichier = time.ctime(os.path.getctime(PREPROCESSED_DATA_PATH))
    #         st.write("Date de creation du dernier fichier : " + str(date_creation_fichier))
    #         predict_btn = st.button('Generer predictions')
    #         if predict_btn:
    #             with st.spinner("Operation en cours..."):
    #                 current_data = pd.read_csv(PREPROCESSED_DATA_PATH)
    #                 current_data = current_data.drop(columns=['SK_ID_CURR'])
    #                 # pred_results = make_predictions_lot(current_data.values, current_data.columns, MODEL_URI)
    #                 st.write(pred_results)
    #             st.success("Un jeu de donnees predites a ete genere")
    #     else:
    #         st.markdown("Aucun jeu de donnees n'a encore ete charge. Utilisez la navigation pour charger un fichier")

    else:
        st.markdown("This page is not implemented yet :no_entry_sign:")


if __name__ == '__main__':
    main()
