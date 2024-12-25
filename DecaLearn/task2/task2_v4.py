import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.utils import to_categorical

# Carrega a base de dados a partir de seu caminho
data = np.load('dataset.npz', allow_pickle=True)
X_test = data['X_test']
X_train = data['X_train']
y_test = data['y_test']
y_train = data['y_train']

# Convert labels to categorical
num_classes = len(np.unique(y_train))
y_train_categorical = to_categorical(y_train, num_classes)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

#Model
print('model')
initializer = keras.initializers.GlorotNormal(seed=12227)

inputs_layer = keras.layers.Input(shape=(X_test_sc.shape[1],))

H = keras.layers.Dense(units=64, activation='relu')(inputs_layer)
H = keras.layers.Dense(units=32, activation='relu')(H)
outputs_layer = keras.layers.Dense(units=num_classes, activation='softmax')(H)

model = keras.models.Model(inputs=inputs_layer, outputs=outputs_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_sc, y_train_categorical, epochs=50, batch_size=32)

y_pred_prob = model.predict(X_test_sc)

y_pred_classes = np.argmax(y_pred_prob, axis=1)

# Gera um DataFrame no formato esperado da submissão
num_samples = X_test.shape[0]
submission_df = pd.DataFrame({
    'ID': np.arange(1, num_samples + 1),
    'Prediction': y_pred_classes
})

# Salva o arquivo CSV no diretório atual
submission_df.to_csv('submission_task2_v4.csv', index=False)