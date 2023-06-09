import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


###########################################################################################
# ADDITIONAL METRICS
###########################################################################################
def compute_F2(y_true, y_pred):
    faux_negs = np.sum((y_true == 1) & (y_pred == 0))
    faux_pos = np.sum((y_true == 0) & (y_pred == 1))
    true_pos = np.sum((y_true == 1) & (y_pred == 1))
    # true_negs = np.sum((y_true == 0) & (y_pred == 0))
    f2_score = true_pos / (true_pos + 0.2 * faux_pos + 0.8 * faux_negs)
    return f2_score


def compute_F2_custom(y_true, y_pred):
    faux_negs = np.sum((y_true == 1) & (y_pred == 0))
    faux_pos = np.sum((y_true == 0) & (y_pred == 1))
    true_pos = np.sum((y_true == 1) & (y_pred == 1))
    # true_negs = np.sum((y_true == 0) & (y_pred == 0))
    f2_score = true_pos / (true_pos + 0.1 * faux_pos + faux_negs)
    return f2_score


###########################################################################################
# RESULTS VISUALISATION
###########################################################################################
# CLASSIFICATION
def matrice_de_confusion_binaire(y_true, y_pred, complement_titre, path, nomfichier, annot=True, save=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    mat = np.round(confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true'), 2)
    sns.heatmap(mat, annot=annot, fmt='.2f')
    plt.ylabel('Categories reelles')
    plt.xlabel('Categorie predites')
    plt.title('Matrice de confusion ' + complement_titre + ' \n', fontsize=18)
    if save:
        plt.savefig(path + nomfichier)
    plt.show()
    return path + nomfichier


###########################################################################################
# FEATURE IMPORTANCE
###########################################################################################
# LOGREG
def feature_importance_LogisticRefression(estimator, feature_names, n_coefs=40):
    coefficients = estimator.coef_[0]
    feat_names = np.array(feature_names)  # X.columns.tolist()
    resd_df = pd.DataFrame()
    resd_df['feature'] = feat_names
    resd_df['coeflogreg'] = coefficients
    # Sort coefficients in descending order and get the 40 largest coefficients
    top40_coefficients = np.argsort(np.abs(coefficients))[::-1][:n_coefs]
    # Plot the values of the selected coefficients
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top40_coefficients)), coefficients[top40_coefficients])
    plt.xticks(range(len(top40_coefficients)), feat_names[top40_coefficients], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Coefficient Value')
    plt.title(f'Top {n_coefs} Coefficients of Logistic Regression Model')
    plt.show()
    return resd_df


# RANDOM FOREST
def feature_importance_RandomForest(xtrain, modelrf, top=True, size=20):
    feat_imps = pd.DataFrame()
    feat_imps['feature'] = xtrain.columns
    feat_imps['poids'] = modelrf.feature_importances_
    # feat_imps['parametres'] = [BESTTESTDESC for i in range(len(xtrain.columns))]
    if top:
        feat_imps = feat_imps.sort_values(by=['poids'], ascending=False).head(size)
    plt.figure(figsize=(14, 10))
    sns.color_palette("rocket", as_cmap=True)
    sns.barplot(x=feat_imps['poids'], y=feat_imps['feature'], orient='h', palette="rocket")
    # sns.despine(left=True, bottom=True) # Vire les axes
    plt.plot
    # Setting the label for x-axis
    plt.xlabel("FEATURES")
    # Setting the label for y-axis
    plt.ylabel("BETA_i")
    # Setting the title for the graph
    plt.title("Coefficients des features avec le Random Forest")
    plt.xticks(rotation=90)
    plt.show()
    return feat_imps


# XGBoost
def feature_importance_XGBoost(xgb_estimator, feature_names, top=15, importance_type='weight'):
    # Associate feature name to feature number
    feature_names = feature_names
    df_feat_cross = pd.DataFrame(feature_names, columns=['feature_name'])
    df_feat_cross['feature_number'] = [f'f{i}' for i in range(feature_names.shape[0])]

    # Retrieve the feature importance scores
    importance_scores = xgb_estimator.get_booster().get_score(importance_type=importance_type)
    # Convert importance_scores to a DataFrame
    df_importance = pd.DataFrame(list(importance_scores.items()), columns=['feature_number', importance_type])
    # # Sort the DataFrame by weight in descending order
    df_importance = df_importance.sort_values(by=importance_type, ascending=False)
    # Connect the name
    df_importance = df_importance.merge(df_feat_cross, on='feature_number', how='left')
    df_importance = df_importance.head(top)

    plt.figure(figsize=(14, 10))
    sns.color_palette("rocket", as_cmap=True)
    sns.barplot(x=df_importance[importance_type], y=df_importance['feature_name'], orient='h', palette="rocket")
    # sns.despine(left=True, bottom=True) # Vire les axes
    plt.plot
    # Setting the label for x-axis
    plt.xlabel("FEATURES")
    # Setting the label for y-axis
    plt.ylabel(importance_type)
    # Setting the title for the graph
    plt.title(f"Feature Importance XGBoost - Top {top}")
    plt.xticks(rotation=90)
    plt.show()
    return  df_importance
