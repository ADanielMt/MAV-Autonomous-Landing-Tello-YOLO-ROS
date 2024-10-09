import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
#from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error

df_nn = pd.read_csv('qr_height_cleaned.csv')

array = df_nn.values
X = array[:,0:4]
Y = array[:,4:5]

# Normalizar características
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X.values)

# Guardar el escalador
with open('scaler_mlp_v2.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(scaled_features, Y, test_size=0.2, random_state=42)

# Convertir etiquetas a one-hot encoding
num_classes = 13
y_train_categorical = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)

# Modelo clasificador
classifier_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])


# Compilar el modelo clasificador
classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo clasificador
classifier_model.fit(X_train, y_train_categorical, epochs=300, batch_size=32, validation_split=0.2)

# Evaluar el modelo clasificador
y_pred_classifier = classifier_model.predict(X_test)
y_pred_classifier_classes = np.argmax(y_pred_classifier, axis=1)
y_test_classes = np.argmax(y_test_categorical, axis=1)
accuracy = accuracy_score(y_test_classes, y_pred_classifier_classes)
precision = precision_score(y_test_classes, y_pred_classifier_classes, average='macro')
recall = recall_score(y_test_classes, y_pred_classifier_classes, average='macro')
print("Classifier Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

from sklearn.metrics import confusion_matrix  
print(confusion_matrix(y_test_classes, y_pred_classifier_classes))  

# Guardar el modelo entrenado
classifier_model.save('qr_height_mlp_v2.h5')