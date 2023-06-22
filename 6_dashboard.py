import streamlit as st
import pandas as pd
import mbr_kernel as mkn
import os.path
import time
import json
import requests
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from outils_general_810 import list_files_in_folder
from outils_feature_engineering_810 import get_cat_for_obs


PREPROCESSED_DATA_PATH = './st_content/processed_new_data.csv'
ORIGINAL_DATA_PATH = './st_content/original_data.csv'
CUSTOM_CSS = './st_content/style.css'
HEROKU_APP_PREDICTION_URI = "https://oc-ds-p7.herokuapp.com/solvability_prediction"
HEROKU_APP_SHAP_FORCE_URI = "https://oc-ds-p7.herokuapp.com/get_shap_force"
HEROKU_APP_PREPROCESSING_URI = "https://oc-ds-p7.herokuapp.com/preprocess_data"
TOP_FEATURES_FILES = "./st_content/top_features_profiles/"


def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    return


# ------------------------------------------------------------------------------------
# FONCTIONS D'APPELS D'API
# ---------------------------------------------------------------------------------


def predict_with_heroku_app(lst_columns, associated_data):
    dict_customer = {k: v for k, v in zip(lst_columns, associated_data)}
    input_json = json.dumps(dict_customer)
    response = requests.post(HEROKU_APP_PREDICTION_URI, data=input_json)
    return response.text


def get_shap_force(lst_columns, associated_data):
    dict_customer = {k: v for k, v in zip(lst_columns, associated_data)}
    input_json = json.dumps(dict_customer)
    response = requests.post(HEROKU_APP_SHAP_FORCE_URI, data=input_json)
    return json.loads(response.text)


def get_preprocessed_data(lst_columns, associated_data):
    dict_customer = {k: v for k, v in zip(lst_columns, associated_data)}
    input_json = json.dumps(dict_customer)
    response = requests.post(HEROKU_APP_PREPROCESSING_URI, data=input_json)
    return json.loads(response.text)


# ---------------------------------------------------------------------------------
# FONCTIONS DE TRAITEMENTS LOCAUX
# ---------------------------------------------------------------------------------
def dict_to_pd(data):
    df = pd.DataFrame.from_dict(data, orient='index', columns=['values'])
    df = df.reset_index()
    df.columns = ['features', 'values']
    return df


def get_radar_values(obs, path_to_qcuts_df):
    exceeds_train = False
    output_dict = {}
    lst_files = list_files_in_folder(path_to_qcuts_df)
    for file in lst_files:
        feat_name = file.split('.')[0]
        qcut_df = pd.read_csv(path_to_qcuts_df + file)
        cat, ex_cat = get_cat_for_obs(obs, feat_name, qcut_df)
        output_dict[feat_name] = cat
        if ex_cat:
            exceeds_train = True

    if exceeds_train:
        output_dict['ExceedsKnownData'] = 1
    else:
        output_dict['ExceedsKnownData'] = 0
    return output_dict



def get_rad_val(application, ref_folder=TOP_FEATURES_FILES):
    dict_radar_values = get_radar_values(application, ref_folder)
    return dict_radar_values


# ---------------------------------------------------------------------------------
# FONCTIONS D'AFFICHAGE DE GRAPHS
# ---------------------------------------------------------------------------------


def plot_top_x_bar(df_input, colname_name, colvalue_name, top=10):
    df_sorted = df_input
    df_sorted['abs_val'] = np.abs(df_sorted[colvalue_name])
    df_sorted = df_input.sort_values(by=['abs_val'], ascending=False).head(top)
    # Reinverting for display purposes
    df_sorted = df_sorted.sort_values(by=['abs_val'], ascending=True)
    fig = px.bar(df_sorted, x=colvalue_name, y=colname_name, orientation='h')

    # Resize text
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=18,  # Set the font size here
            color="RebeccaPurple"
        )
    )
    return fig


def visualize_radar_chart(data):
    tmp_data = data
    tmp_data.pop('ExceedsKnownData', None)

    # Extract the keys and values from the dictionary
    keys = list(tmp_data.keys())
    values = list(tmp_data.values())

    # Calculate the angle for each value
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]  # Repeat the first angle to close the circle

    # Duplicate the first value to make both lists the same length
    values += values[:1]

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
    ax.fill(angles, values, color='skyblue', alpha=0.7)
    # ax.grid(False)
    ax.set_xticks(angles[:-1], )
    ax.set_xticklabels(keys)
    ax.set_yticks([])  # Remove radial labels

    # ax.spines['polar'].set_color('none')  # Set radial grid lines color to transparent
    # ax.set_rlabel_position(-22.5)
    ax.set_ylim(0, max(values)+1)
    ax.spines['polar'].set_visible(False)  # Hide the radial grid lines

    # Customize the grid lines
    ax.xaxis.grid(color='lightgray', linestyle='--', alpha=0.5)
    ax.yaxis.grid(color='lightgray', linestyle='--', alpha=0.5)

    # ax.yaxis.grid(False)

    # Add values as labels
    # for angle, value in zip(angles[:-1], values[:-1]):
    #     ax.text(angle, value + 0.5, str(value), ha='center', va='center')

    # Display the radar chart
    st.pyplot(fig)
    return


# ---------------------------------------------------------------------------------
# FONCTIONS AFFICHAGE PAGE WEB
# ---------------------------------------------------------------------------------


def main():
    local_css(CUSTOM_CSS)

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
            if sk_id_combo is not None :
                liste_colonnes = current_data.drop(columns='SK_ID_CURR').columns.tolist()
                data_client = current_data[current_data['SK_ID_CURR'] == sk_id_combo].drop(
                    columns=['SK_ID_CURR']).values

                # AFFICHAGE RADAR
                pp_data = get_preprocessed_data(lst_columns=liste_colonnes,
                                                associated_data=data_client.tolist()[0])
                rad_values = get_rad_val(application=pp_data)

                visualize_radar_chart(rad_values)

                predict_btn = st.button('Etudier dossier')
                if predict_btn:
                    if sk_id_combo is not None:


                        # AFFICHAGE PREDICTION
                        prediction = predict_with_heroku_app(lst_columns=liste_colonnes,
                                                             associated_data=data_client.tolist()[0])

                        if prediction == '0':
                            text_result = "<span class='ok_customer'> OK </span>"
                            st.markdown(text_result, unsafe_allow_html=True)
                        else:
                            text_result = "<span class='nok_customer'> NOT OK </span>"
                            st.markdown(text_result, unsafe_allow_html=True)

                        # AFFICHAGE SHAP FORCE
                        st.write("TOP 15 Criteria")
                        shap_force = get_shap_force(lst_columns=liste_colonnes,
                                                    associated_data=data_client.tolist()[0])

                        df_shape_force = dict_to_pd(shap_force)

                        fig = plot_top_x_bar(df_shape_force, colname_name='features', colvalue_name='values')
                        st.write(fig)
                    else:
                        st.markdown(
                            "ID client manquant.")
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
