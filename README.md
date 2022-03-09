# Catégorisez automatiquement des questions
## Context
Dans le cadre de la formation OpenClassRooms, ce projet consiste à créer un outil de recommandation de tags pour les utilisateurs du site stackoverflow.
- La liste de tags est dans le fichier /models/tags_list.
- Le modèle utilisé est Tf-idf + OneVsRest(LogisticRegressor).

## API
Afin de tester le modèle selectionné, un point d'entrée API est proposé : https://projet-ocl-tags.herokuapp.com/

## Fonctionnement
### Nettoyage
Dans le fichier /ressources/prediction_model.py, les données "question" et "details" sont nettoyées :
- suppression des blocs de code
- suppression des balises HTML
- suppression des caractères unicode
- passage en minuscule
- suppression des nombres
- suppression des caractères de ponctuation
- tokenisation
- suppression des stop words
- conservation des noms
- lemmatisation

### Prediction
Dans le fichier /ressources/prediction_model.py, le modèle selectionné est appliqué et le résultat est affiché. Si aucun tag n'est prédit, un message "No tag found..." apparait.

## Commentaires
Le cas de données invalides n'est pas pris en compte, le programme aura en sortie le message "No tag found..."
