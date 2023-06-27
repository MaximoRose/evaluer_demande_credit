# ocdsp7_dashboard_sl
Sous-repo du projet OC-DS-P07, dédié à l'hébergement du dashboard interactif.

L'ensemble des fichiers présents provient du repo du projet principal :
1. 6_dashboard.py : est le code du Dashboard
2. mbr_kernel.py : contient les fonctions de prétraitement des données métiers spécifiques à ce projet.
3. outils_feature_engineering_810.py : contient des fonctions de prétraitement générales. En l'occurrence on utilise dans le dashboard une fonction qui permet de positionner une observation parmi les quartiles d'une variables, par rapport au jeu de données de test (pour le radar chart)
4. outils_general_810.py : contient des fonctions dont j'ai régulièrement besoin. En l'occurrence, ici, lister les fichiers d'un dossier.
5. Le dossier st_content : contient des éléments nécessaires à l'interface graphique : un fichier CSS, un sample data et un bad sample data pour le test. Par ailleurs un dossier top_features_profiles contient pour les top features globale du modèle, les intervals définissant les quartiles dans le jeu de données de d'entraînement.  (pour le radar chart)

Les appels à l'API sont de plusieurs ordres : 
1. Préprocessing de données (exécution du pipeline du modèle sans l'estimateur)
2. Prédiction de la probabilité d'être 0 ou 1
3. Prédiction du résultat (0 ou 1) (CSI : 1 et 2 pourrait à l'avenir ne constituer qu'une seule fonction pour 2 et 3)
3. Récupérer les coefficient de SHAP Force liés à une décision.