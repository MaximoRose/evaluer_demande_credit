# Evaluer des demandes de prets.

## Mission.
À partir des données fournies par un organisme de Crédit, nous allons concevoir un modèle capable de prédire la capacité d'une personne à rembourser son credit.

- Le suivi des tests des modèles sera realisé dans MLFlow. Les meilleurs modèles seront sauvegardés dans le registre.
- L'évaluation des performances sera réalisée avec le F2-score, afin de respecter les enjeux métiers.
- Le Data drift éventuel sera évalué avec Evidently et des axes de compensation seront proposés.
- Nous déploierons le modèle choisi sur Heroku pour le rendre accessible via API.
- Nous développerons une interface graphique pour les utilisateurs et utilisatrices via Streamlit. Nous utiliserons des modèles d'inférence pour expliquer les résultats retournés par notre modèle principal.
- GitHub Actions permettra d'automatiser les tests de cohérences entre l'API et le tableau de bord.

La structure de données est la suivante :

![structure des donnees](https://maximorose.eu/datascience_gh_ress/p7_structure_donnees.png)


## Remarque RGPD & Feature Engineering.
La quantité de données personnelles soumises par l'organisme de crédits est immense... Il y a le sexe, la situation familiale, le nombre d'enfants, l'employeur, les accompagnants de la personne lors du rendez-vous avec le conseiller, le type de logement, etc.

Cela pose plusieurs problèmes :
1. D'un point de vue de la réglementation européenne, on est sensé collecter et traiter seulement les données nécessaires à la délivrance du service mis en place. Or, l'étude du coefficient de corrélation de ces variables à notre cible montre qu'elles sont pour la plupart peu utiles.
2. Concernant celles qui améliorent la précision des prédictions, comme le sexe, elles favorisent un apprentissage discriminant : Le modèle va apprendre de nos biais sociétaux pour les reproduire.

J'ai donc fait le choix de retirer toute donnée socio-démographique pour que le modèle n'apprenne que sur des données d'historique bancaire.

La précision des modèles sera légèrement réduite, mais leur rapidité augmentée, puisque nous réduisons ainsi la dimensionnalité.

Par ailleurs, de nombreuses variables sont corrélées les unes aux autres, et d'autres présentent une très grande part de valeurs manquantes. Ainsi, dans un premier temps, on fera le choix d'enlever toutes ces variables qui troubleraient l'explicabilité des résultats.

Notre _feature engineering_ reposera donc sur les principes suivants :

![feature engineering projet](https://maximorose.eu/datascience_gh_ress/p7_FE_custom.png)


__Attention :__ 
- Pour pouvoir exécuter les codes présents dans ce repo, il est nécessaire de créer d'abord un dossier "input_data" dans lequel déposer les fichiers .csv du projet Kaggle concerné. Ces fichiers étant très volumineux, il nous était impossible de les référencer dans GitHub.
- Pour ces mêmes raisons, nous ne pourrons mettre à disposition l'ensemble des tests réalisés via MLFlow. Nous  nous contenterons donc de laisser seulement quelques éléments pertinents qui permettent de constater les enregistrements spécifiques pour les GridSearch et les "Best Models"


## Architecture du projet.

La production du projet se divise en 3 _repositories_, mais le présent _repository_ regroupe l'ensemble des codes nécessaires.

- L'interface graphique est accessible sur streamlit mais s'appuie sur un repository dédié.
- De même, l'API est hébergée sur Heroku, mais s'appuie, elle-aussi, sur un _repository_ dédié.

![architecture projet globale](https://maximorose.eu/datascience_gh_ress/p7_archi_global.png)



## Les dossiers
- __./.github/workflows/ : Contient le fichier .yml qui définit la séquence d'actions à lancer lorsque le repo est poussé dans GitHub__
- ./.pytest_cache/ : Dossier automatiquement généré par pytest, mais non inclus dans le repo GitHub
- __./.streamlit/ : contient un fichier de config .toml qui peut permettre de modifier l'interface graphique de streamlit.__ 
- __./api_content/ : copie du contenu du repo dédié à l'API utilisé par Heroku__
- __./dashboard_content/ : copie du contenu du repo dédié au DashBoard et utilisée par Streamlit cloud__
- __./evidently_datadrift/ : rapports de datadrift générés par evidently__
- ./kaggle_kernels/ : les kernels kaggle que j'ai étudié pour initier le projet
- ./old_workspace/ : des fichiers issus d'anciens tests que j'ai souhaité conserver. La version de 4_Feature_Importances_in_XGBoost.ipynb qu'on y trouve met en évidence le problème lié aux EXT_SOURCE variables (_data leakage_)
- ./processed_data/ :  des traces de différents tests dont j'ai sauvegardé certaines données pour pouvoir les réutiliser d'un jour à l'autre sans avoir à ré-effectuer les fonctions associées.
- ./st_content/ : un répertoire dédié à l'interface web du tableau de bord. Contient aussi bien des fichiers nécessaires à son fonctionnement, que des fichiers qu'elle génère.
- __./unitest/ : les fichiers de tests unitaires associés à pytest__

## Jupyter principaux
Ce _repository_ est le dossier de travail principal pour le projet. Il contient une série de Jupyter numérotés, dont l'ordre correspond aux étapes suivies pour l'amélioration continue du modèle :

- 0 - Kernel Kaggle Analysis : repartir des kernel Kaggle pour revoir les features les plus importantes 
- 1 - Analyse des corrélations : pour étudier plus en détails les corrélations entre les variables et la target, afin de pouvoir les filtrer.
- 2 - Dummy Classifier : pour disposer d'une référence en termes de métriques de qualité du modèle.
- 3 - Logisitic Regression : Le fichier d'exécution et d'analyse des performances de la régression logistique.
- 3 - Random Forest : De même pour la Random Forest
- 3 - XGBoost : idem pour XGBoost.
- 4 - Feature importance in Models : est une analyse de la _feature importance_ globale pour les modèles non retenus, mais qui ont servi la construction du projet (LogReg & Random Forest)
- 4 - Feature importance XGBoost : est l'analyse tant globale que locale des _feature importances_ pour le modèle XGBoost en production.
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


