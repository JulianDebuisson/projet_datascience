import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from math import ceil
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

# Importation des données

file = 'car_insurance.csv'
df = pd.read_csv(file)

print(df.head())
print()

# Examen des données

print("Taille du jeu de données : ", df.shape)
print("Types de données : \n", df.dtypes, "\n")
print("Valeurs manquantes : \n", df.isnull().sum(), "\n")

# Suppression des colonnes inutiles

df.drop(columns=['id', 'age'], inplace=True)

# suppression des données avec des valeurs manquantes

#df.dropna(inplace=True)
#print("Valeurs manquantes après suppression : \n", df.isnull().sum(), "\n")

# Remplacement des valeurs nulles dans les colonnes "credit_score" et "annual_mileage" par la moyenne de ces colonnes
df['credit_score'].fillna(df['credit_score'].median(), inplace=True)
df['annual_mileage'].fillna(df['annual_mileage'].median(), inplace=True)
  
print(df.shape)

columns_to_plot = [
    'speeding_violations', 'duis', 'past_accidents', 'age', 'gender',
    'credit_score', 'vehicle_ownership', 'married', 'children', 'annual_mileage'
]

# Affichage des histogrammes dans une seule fenêtre matplotlib
num_columns = len(columns_to_plot)
num_rows = ceil(num_columns / 3)  # 3 colonnes par ligne

fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
axes = axes.flatten()

for i, column in enumerate(columns_to_plot):
    if column in df.columns:
        df[column].hist(bins=20, edgecolor='black', ax=axes[i])
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

# Supprimer les axes inutilisés si le nombre de colonnes n'est pas un multiple de 3
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Modification des données abérrantes

# Remplacement des valeurs supérieures à 50 dans 'speeding_violations' par la moyenne de la colonne
print("Modification des valeurs supérieures à 50 dans 'speeding_violations' par la moyenne de la colonne")

mean_speeding_violations = df['speeding_violations'][df['speeding_violations'] <= 50].mean()
df.loc[df['speeding_violations'] > 50, 'speeding_violations'] = ceil(mean_speeding_violations)

print("\n\nMoyenne utilisée pour remplacement : ", ceil(mean_speeding_violations))
print(df['speeding_violations'].describe())

print("Modification des valeurs supérieures à 1 dans 'children' par la valeur 1")

df.loc[df['children'] > 1, 'children'] = 1

print("\n\nMoyenne utilisée pour remplacement : ", 1)
print(df['children'].describe())


# Modification des données quantitatives en variables numériques
# Transformation des données quantitatives en variables numériques
columns_to_encode = ['income', 'vehicle_year', 'vehicle_type', 'driving_experience', 'education']
label_encoder = LabelEncoder()

for column in columns_to_encode:
    if column in df.columns:
        df[column] = label_encoder.fit_transform(df[column])

print("Données transformées avec LabelEncoder :")
print(df.head())

# Normalisation des colonnes spécifiées
columns_to_normalize = ['duis', 'speeding_violations', 'annual_mileage', 'income', 'vehicle_year', 'vehicle_type', 'driving_experience', 'past_accidents', 'education', 'postal_code']
scaler = StandardScaler()

df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

print("Données normalisées :")
print(df.head())

# Calcul du coefficient de corrélation entre chaque couple de variables numériques
correlation_matrix = df.corr()
print("Matrice de corrélation :")
print(correlation_matrix)
# Affichage de la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(12, 10))
plt.title("Matrice de corrélation")
heatmap = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(heatmap)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.tight_layout()
plt.show()

# Affichage des scatter plots pour visualiser les corrélations entre les colonnes
scatter_columns = ['speeding_violations', 'duis', 'past_accidents', 'gender',
    'credit_score', 'vehicle_ownership', 'married', 'children', 'annual_mileage']
scatter_matrix(df[scatter_columns], figsize=(12, 8), diagonal='kde', alpha=0.7)
plt.suptitle("Scatter Matrix des colonnes sélectionnées")
plt.show()

# Séparation des données en variables explicatives (X) et variable cible (y)
X = df.drop(columns=['outcome'])  # Remplacez 'outcome' par le nom de votre colonne cible
y = df['outcome']  # Remplacez 'outcome' par le nom de votre colonne cible

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Taille de l'ensemble d'entraînement :", X_train.shape)
print("Taille de l'ensemble de test :", X_test.shape)

# Entraînement du modèle de régression logistique
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
print("\n\nModèle de régression logistique entraîné.")

# Prédiction sur l'ensemble de test
print("Précision du modèle : ", model.score(X_test, y_test))

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)
print("\nPrécision F1 : ", f1_score(y_test, y_pred))
print("Précision Confusion : ", confusion_matrix(y_test, y_pred))
print("Précision Précision : ", precision_score(y_test, y_pred))
print("Précision Rappel : ", recall_score(y_test, y_pred))

# Validation croisée pour évaluer le modèle
cv_scores = cross_val_score(model, X, y, cv=5)
print("Scores de validation croisée :", cv_scores)
print("Score moyen de validation croisée :", cv_scores.mean())

# Entraînement du modèle Perceptron
perceptron_model = Perceptron(max_iter=10000)
perceptron_model.fit(X_train, y_train)
print("\n\nModèle Perceptron entraîné.")

# Prédiction sur l'ensemble de test avec le Perceptron
perceptron_accuracy = perceptron_model.score(X_test, y_test)
print("Précision du modèle Perceptron : ", perceptron_accuracy)


# Validation croisée pour le Perceptron
perceptron_cv_scores = cross_val_score(perceptron_model, X, y, cv=5)
print("Scores de validation croisée pour le Perceptron :", perceptron_cv_scores)
print("Score moyen de validation croisée pour le Perceptron :", perceptron_cv_scores.mean())

# Entraînement du modèle K plus proche voisin
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
print("\n\nModèle K plus proche voisin entraîné.")

# Prédiction sur l'ensemble de test avec le modèle KNN
knn_accuracy = knn_model.score(X_test, y_test)
print("Précision du modèle K plus proche voisin : ", knn_accuracy)

# Validation croisée pour le modèle KNN
knn_cv_scores = cross_val_score(knn_model, X, y, cv=5)
print("Scores de validation croisée pour le modèle KNN :", knn_cv_scores)
print("Score moyen de validation croisée pour le modèle KNN :", knn_cv_scores.mean())

scores = {
    model : cv_scores.mean(),
    perceptron_model : perceptron_cv_scores.mean(),
    knn_model : knn_cv_scores.mean()
}

# Enregistrement du meilleur modèle
best_model = max(scores, key=scores.get)
print("Meilleur modèle : ", best_model)
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)