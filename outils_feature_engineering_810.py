import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

DEFAULT_CATEGORY_MEAN_FREQ = 0.03
DEFAULT_MAX_PERCENTAGE_NAN_VAL = 0.3


#################################################################################
# ---------------------------------- TRANSFORM COLUMNS
#################################################################################


def one_hot_encoder(df, categorical_columns=[], nan_as_category=True, treshold=DEFAULT_CATEGORY_MEAN_FREQ):
    original_columns = list(df.columns)
    if categorical_columns == []:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        # print("Cat col in OHE : ", categorical_columns)
    # MBR TEST SANS MODIFIER LES LOW CARDINALS
    df_res = regroup_low_cardinals(df, categorical_columns=categorical_columns,
                                   treshold=treshold)  # Ajout MBR pour eviter overfit
    df_res = pd.get_dummies(df_res, columns=categorical_columns, dummy_na=nan_as_category)
    # df_res = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df_res.columns if c not in original_columns]
    return df_res, new_columns


def regroup_low_cardinals(df, categorical_columns=[], treshold=DEFAULT_CATEGORY_MEAN_FREQ):
    # print("Regrouping low cards")
    # print("Colonne a processer : ", categorical_columns)
    # print("treshold : ", treshold)
    # print(categorical_columns)
    for col in categorical_columns:
        tmp_df = df[col].value_counts(normalize=True).reset_index()
        values_to_unify = tmp_df[tmp_df[col] < treshold]['index'].values.tolist()
        df.loc[df[col].isin(values_to_unify), col] = col + "_divers"
    return df


# TRAITEMENT DES CELLULES VIDES
# ----------
def remove_too_nany_observations(df, treshold=0.3, target_in_df=True):
    res_df = df
    nb_features = df.shape[1]
    if target_in_df:
        nb_features = df.shape[1] - 1
    print("Forme initiale du Dataframe : ", res_df.shape)
    res_df.loc[:, 'taux_nan'] = res_df.isna().sum(axis=1) / nb_features
    res_df = res_df[res_df['taux_nan'] < treshold]
    res_df.drop(columns=['taux_nan'], inplace=True)
    print("Forme du Dataframe apres traitement : ", res_df.shape)
    return res_df


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def remove_too_nany_columns(df, treshold=DEFAULT_MAX_PERCENTAGE_NAN_VAL):
    """
    Supprime les colonnes dont la part de NaN est superieure a un certains pourcentage
    :param df:
    :param treshold:
    :return:
    """
    percent_treshold = treshold * 100
    print("Forme avant traitement : ", df.shape)
    missing_values = missing_values_table(df=df)
    not_too_nany_cols = [col for col in df.columns if
                         col not in missing_values[
                             missing_values['% of Total Values'] > percent_treshold].index.tolist()]
    print(
        f"Suppression des {df.shape[1] - len(not_too_nany_cols)} colonnes avec plus de {percent_treshold}% de valeurs manquantes")
    df_moderate_bias = df[not_too_nany_cols]
    print(f"Dataset sans les features vides a plus de {percent_treshold}% : ", df_moderate_bias.shape)
    return df_moderate_bias


def replace_infinite_by_nan(df, list_new_columns):
    for col in list_new_columns:
        df.loc[~np.isfinite(df[col]), col] = np.nan
    return df


#################################################################################
# ---------------------------------- ANALYSE CORRELATIONS
#################################################################################
def get_highly_correlated_features(df, threshold):
    correlation_matrix = df.corr()
    highly_correlated = correlation_matrix.abs() > threshold

    # Create an empty dataframe to store the results
    correlation_results = pd.DataFrame(columns=['Feature 1', 'Feature 2', 'Correlation'])

    # Iterate over the highly correlated features
    for feature1 in highly_correlated:
        for feature2 in highly_correlated[feature1][highly_correlated[feature1]].index:
            # Exclude self-correlations & Correlation already listed in reverse order
            if (feature1 != feature2) & (correlation_results[(correlation_results['Feature 1'] == feature2) & (
                    correlation_results['Feature 2'] == feature1)].shape[0] == 0):
                correlation = correlation_matrix.loc[feature1, feature2]
                correlation_results = correlation_results.append({
                    'Feature 1': feature1,
                    'Feature 2': feature2,
                    'Correlation': correlation
                }, ignore_index=True)

    # Display the dataframe with highly correlated features
    return correlation_results


def display_barchart_bivar_correlation(df_poids, col_nom='feature', col_corr='poids', variable_ref='TARGET',
                                       coef='Pearson'):
    feat_corrs = df_poids
    plt.figure(figsize=(14, 10))
    sns.color_palette("rocket", as_cmap=True)
    sns.barplot(x=feat_corrs[col_corr], y=feat_corrs[col_nom], orient='h', palette="rocket")
    # sns.despine(left=True, bottom=True) # Vire les axes
    plt.plot
    # Setting the label for x-axis
    plt.xlabel("FEATURES")
    # Setting the label for y-axis
    plt.ylabel("Coef / " + variable_ref)
    # Setting the title for the graph
    plt.title("Coefficients de correlation (" + coef + ")")
    plt.xticks(rotation=90)
    plt.show()
    return


# ------------------ VISUALISATION
def display_corrmat(corr, annot=True):
    sns.set(context="paper", font_scale=1.2)
    # Mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Plot
    f, ax = plt.subplots(figsize=(12, 12))
    f.text(0.45, 0.93, "Coefficients de correlation de Pearson", ha='center', fontsize=18)
    sns.heatmap(corr, mask=mask, square=True, linewidths=0.01, cmap="coolwarm", annot=annot, vmin=-1, vmax=1,
                center=0)
    plt.tight_layout()
    return


#################################################################################
# ---------------------------------- OPTIMISATION
#################################################################################
def reduce_memory(df):
    """
    Reduce memory usage of a dataframe by setting data types.
    :param df:
    :return:
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    # print('Initial df memory usage is {:.2f} MB for {} columns'
    #       .format(start_mem, len(df.columns)))

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            cmin = df[col].min()
            cmax = df[col].max()
            if str(col_type)[:3] == 'int':
                # Can use unsigned int here too
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    memory_reduction = 100 * (start_mem - end_mem) / start_mem
    # print('Final memory usage is: {:.2f} MB - decreased by {:.1f}%'.format(end_mem, memory_reduction))
    return df


#################################################################################
# ---------------------------------- WORKING WITH QCUTS
#################################################################################
def get_qcut_of_columns(idf, colonne, nbcuts=4):
    """ Pour une colonne, retourne un dataframe, trie par ordre croissant contenant les categories et leur limites suite a un qcut de taille nbcuts
    colonne_qcuts : Les intervals pandas definis par la fonction qcuts
    left_value : La valeur min de l'interval
    right_value : La valeur max de l'interval
    """
    res_df = pd.DataFrame()
    try:
        res_df['colonne_qcuts'] = pd.qcut(idf[colonne], nbcuts).unique()
        res_df['left_value'] = res_df.colonne_qcuts.map(lambda x: x.left)
        res_df['right_value'] = res_df.colonne_qcuts.map(lambda x: x.right)
        res_df = res_df.sort_values(by=['left_value'], ascending=True)
    except:
        print("Impossible de proceder au qcut pour la colonne : ", colonne)
    return res_df


def get_cat_for_obs(obs, colonne, qcuts_df):
    resulting_cat = 1
    exceeded_limits = False
    try:
        current_value = obs[colonne]
    except:
        resulting_cat = np.nan
        exceeded_limits = True
        return resulting_cat, exceeded_limits
    cnt = 1
    for (borne_inf, borne_sup) in zip(qcuts_df['left_value'].values.tolist(), qcuts_df['right_value'].values.tolist()):
        if (cnt == 1) & (current_value < borne_inf):
            resulting_cat = 1
            exceeded_limits = True
            break
        elif (cnt == (qcuts_df.shape[0])) & (current_value > borne_sup):
            resulting_cat = qcuts_df.shape[0]
            exceeded_limits = True
            break
        elif (current_value > borne_inf) & (current_value <= borne_sup):
            resulting_cat = cnt
            break
        else:
            cnt += 1
    return resulting_cat, exceeded_limits


def list_files_in_folder(folder_path):
    files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            files.append(file_name)
    return files


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
