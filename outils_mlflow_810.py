import mlflow
import pandas as pd
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import recall_score, roc_curve, auc
import outils_classification_810 as oc810

DOSSIER_TMP = './'


def save_grid_results(list_of_image, grid_estimator, ini_params="", path_to_csv=None):
    trimmed_params = []
    if ini_params!="":
        lst_ini_param = ini_params.split(',')
        trimmed_params = [elem.replace(' ', '') for elem in lst_ini_param]
    with mlflow.start_run():
        mlflow.log_param("grid_estimator", grid_estimator)
        for elem in trimmed_params :
            mlflow.log_param(f"INI_{elem.split('=')[0]}", elem.split('=')[-1])
        for image in list_of_image:
            titre = image.split('/')[-1].split('.')[0]
            mlflow.log_artifact(image, titre)
        if path_to_csv:
            img_path = "."+path_to_csv[1:].split('.')[0]+".png"
            mlflow.log_artifact(img_path, "associated_models")
            mlflow.log_artifact(path_to_csv, "associated_models")
    return


def classification_binaire_save_results(xglob, model,
                                 xtrain, ytrain,
                                 xtest, ytest,
                                 nom_model="mon_modele",
                                 estimator="",
                                 tmp_folder=DOSSIER_TMP):
    """Enregistre le test d'un model simple dans MLFlow"""
    df_x_train = pd.DataFrame(xtrain, columns=xglob.columns)

    with mlflow.start_run():
        signature = infer_signature(df_x_train, model.predict(xtrain))
        mlflow.sklearn.log_model(model, nom_model, signature=signature)

        mlflow.log_param("Nb features", len(xglob.columns.tolist()))
        mlflow.log_param("imputer", str(model['imputer']))
        mlflow.log_param("scaler", str(model['scaler']))
        if estimator == "":
            mlflow.log_param("estimator", str(model['estimator']))
        else:
            mlflow.log_param("estimator", estimator)
        mlflow.log_param("over-sampler", str(model['over']))
        mlflow.log_param("under-sampler", str(model['under']))
        y_pred = model.predict(xtest)
        y_train_pred = model.predict(xtrain)

        # -------------------------- LOG RESULTS ------------------
        # Generation des resultats :
        # TRAIN RESULTS
        fpr, tpr, thresholds = roc_curve(ytrain, y_train_pred)
        train_auc_score = auc(fpr, tpr)
        train_F2_score = oc810.compute_F2(y_true=ytrain, y_pred=y_train_pred)
        train_F2c_score = oc810.compute_F2_custom(y_true=ytrain, y_pred=y_train_pred)
        train_rappel = recall_score(y_true=ytrain, y_pred=y_train_pred)
        mlflow.log_metric("TRAIN_F2", train_F2_score)
        mlflow.log_metric("TRAIN_F2C", train_F2c_score)
        mlflow.log_metric("TRAIN_AUC", train_auc_score)
        mlflow.log_metric("TRAIN_Recall", train_rappel)
        # Sauvegarde Confusion Test matrix
        train_img_mtrx = oc810.matrice_de_confusion_binaire(y_true=ytrain, y_pred=y_train_pred, complement_titre=" de Train",
                                                      path=tmp_folder, nomfichier="model_train_cm.png")
        mlflow.log_artifact(train_img_mtrx, "confusion_matrix")

        # TEST RESULTS
        fpr, tpr, thresholds = roc_curve(ytest, y_pred)
        test_auc_score = auc(fpr, tpr)
        test_F2_score = oc810.compute_F2(y_true=ytest, y_pred=y_pred)
        test_F2c_score = oc810.compute_F2_custom(y_true=ytest, y_pred=y_pred)
        test_rappel = recall_score(y_true=ytest, y_pred=y_pred)
        mlflow.log_metric("TEST_F2", test_F2_score)
        mlflow.log_metric("TEST_F2C", test_F2c_score)
        mlflow.log_metric("TEST_AUC", test_auc_score)
        mlflow.log_metric("TEST_Recall", test_rappel)
        # Sauvegarde Confusion Test matrix
        test_img_mtrx = oc810.matrice_de_confusion_binaire(y_true=ytest, y_pred=y_pred, complement_titre=" de Test",
                                                path=tmp_folder, nomfichier="model_test_cm.png")
        mlflow.log_artifact(test_img_mtrx, "confusion_matrix")
    return
