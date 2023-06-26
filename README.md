# Evaluer des demandes de prets.

Attention : Pour pouvoir exécuter les codes présents dans ce repo, il est nécessaire de créer d'abord un dossier "input_data" dans lequel déposer les fichiers .csv du projet Kaggle. Ces fichiers étant très volumineux, il nous était impossible de les référencer dans GitHub ou de les déposer sur la plateforme OC.


## Description générale des fichiers et dossiers
### Boites a outils
- outils_feature_engineering_810.py : des fonctions utiles au feature engineering (analyses de corrélations, encodage des features catégorielles, étude des valeurs manquantes, etc.)
- outils_grid_search_810.py : des fonctions dédiées à l'analyse de résultats des grid search
- outils_classification_810.py : des fonctions dédiées à l'analyse de modèles de classification
- outils_mlflow_810.py : des fonctions dédiées à l'enregistrement d'informations dans ML-Flow

### Preprocessing des donnes custom
- mbr_kernel.py : Ensemble de fonctions de preprocessing des donnees, i.e feature engineering, cleaning, etc.

### Analyse et modelisations

### Visualisations clients
- 6_dashboard.py

### Test divers
Tous les fichiers commencant par "wip_"



