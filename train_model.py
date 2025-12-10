import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import tree
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
seoul_bike_sharing_demand = fetch_ucirepo(id=560) 
  
# data (as pandas dataframes) 
X = seoul_bike_sharing_demand.data.features 
y = X["Rented Bike Count"]
X = X.drop(columns=["Rented Bike Count", "Date"])

#--- 2. One-Hot-Encoding für kategorische Variablen ---
X = pd.get_dummies(X, drop_first=True)

# Boolesche Spalten in 0/1 umwandeln
X = X.astype(int)

# Spaltennamen säubern
X.columns = [c.replace(" ", "_") for c in X.columns]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% als Testdaten
    random_state=42, #seed
)

model = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")

# Mittelwert der Trainingsdaten
y_train_mean = y_train.mean()

# Vorhersage für Testdaten = immer Mittelwert
y_pred_baseline = np.full_like(y_test, y_train_mean)

# MAE der Baseline
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
print(f"Baseline MAE: {mae_baseline:.2f}")

print(f"Train/ Test Bike Count stats:")
print(f"Mean: {y.mean():.1f}, Median: {y.median():.1f}, Max: {y.max()}")


plt.figure(figsize=(20,10))
tree.plot_tree(
    model, 
    filled=True, 
    feature_names=X.columns, 
    fontsize=10
)
plt.show()

joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print(list(X.columns))
