# Evaluer des demandes de prets.

__Attention :__ 
- Pour pouvoir exécuter les codes présents dans ce repo, il est nécessaire de créer d'abord un dossier "input_data" dans lequel déposer les fichiers .csv du projet Kaggle. Ces fichiers étant très volumineux, il nous était impossible de les référencer dans GitHub ou de les déposer sur la plateforme OC.
- Pour ces mêmes raisons, nous ne pourrons vous mettre à disposition l'ensemble des tests réalisés via MLFlow. Nous  nous contenterons donc de laisser seulement quelques éléments pertinent qui permettent de constater les enregistrements spécifiques pour les GridSearch et les "Best Models"


## Les dossiers
- __./.github/workflows/ : Contient le fichier .yml qui définit la séquence d'actions à lancer lorsque le repo est poussé dans GitHub__
- ./.pytest_cache/ : Dossier automatiquement généré par pytest, mais non inclus dans le repo GitHub
- __./.streamlit/ : contient un fichier de config .toml qui peut permettre de modifier l'interface graphique de streamlit.__ 
- __./api_content/ : copie du contenu du repo dédié à l'API utilisé par Heroku__
- __./dashboard_content/ : copie du contenu du repo dédié au DashBoard et utilisée par Streamlit cloud__
- __./evidently_datadrift/ : rapport de datadrift générés par evidently__
- ./kaggle_kernels/ : les kernels kaggle que j'ai étudié pour initier le projet
- ./old_workspace/ : des fichiers issus d'anciens tests que j'ai souhaité conserver. La version de 4_Feature_Importances_in_XGBoost.ipynb qu'on y trouve met en évidence le problème lié aux EXT_SOURCE variables
- ./processed_data/ :  des traces de différents tests dont j'ai sauvegardé certaines données pour pouvoir les réutiliser d'un jour à l'autre sans avoir à ré-effectuer les fonctions associées.
- ./st_content/ : un répertoire dédié à l'interface web du tableau de bord. Contient aussi bien des fichiers nécessaire à son fonctionnement, que des fichiers qu'elle génère.
- __./unitest/ : les fichiers de tests unitaires associés à pytest__

## Jupyter principaux
Ce repository est le dossier de travail principal pour le projet OC_DS_P07. Il contient une série de Jupyter Numérotés, dont l'ordre correspond aux étapes suivies pour l'amélioration continue du modèle :

- 0 - Kernel Kaggle Analysis : repartir des kernel Kaggle pour revoir les features les plus importantes 
- 1 - Analyse des corrélations : pour étudier plus en détails les corrélations entre les variables et à la target, afin de pouvoir les filtrer.
- 2 - Dummy Classifier : pour disposer d'une référence en termes de métriques de qualité du modèle.
- 3 - Logisitic Regression : Le fichier d'exécution et d'analyse des performances de la régression logistique.
- 3 - Random Forest : De même pour la Random Forest
- 3 - XGBoost : idem pour XGBoost.
- 4 - Feature importance in Models : est une analyse de la feature importance globale pour les modèles non retenus, mais qui ont servi la construction du projet (LogReg & Random Forest)
- 4 - Feature importance XGBoost : est l'analyse tant globale que locale des feature importances pour le modèle XGBoost en production.
- 5 - Déploiement model : sert à isoler et produire les éléments nécessaires aux deux interfaces qui seront ensuite à mettre en production : l'interface graphique pour les utilisateurs (GUI / dashboard) ; l'interface de programmation d'application (API)

## Fichiers python principaux
### Le dashboard
Appelé 6_dashboard.py, comme la sixième étape suivant celles ci-dessus, ce fichier peut être lancé via la commande : ``` streamlit run 6_dashboard.py ```

### Preprocessing des donnes custom (.py)
- __mbr_kernel.py :__ Ensemble de fonctions de preprocessing des donnees, i.e feature engineering, cleaning, etc.

### Boites à outils (.py)
Des modules python dédiés à intégrer ma boîte à outils personnelle et utilisés par les fichiers ci-dessus : 
- __outils_classification_810.py :__ des fonctions dédiées à l'analyse de modèles de classification
- __outils_feature_engineering_810.py :__ des fonctions utiles au feature engineering (analyses de corrélations, encodage des features catégorielles, étude des valeurs manquantes, etc.)
- __outils_general_810.py :__ des fonctions en python assez générales
- __outils_grid_search_810.py :__ des fonctions dédiées à l'analyse de résultats des grid search
- __outils_mlflow_810.py :__ des fonctions dédiées à l'enregistrement d'informations dans ML-Flow


