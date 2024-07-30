import numpy as np
import pandas as pd

# Carrega a base de dados a partir de seu caminho
data = np.load('/kaggle/input/pcs5032-decalearn/dataset.npz')
X_test = data['X_test']

# Faz palpites aleatórios
num_samples = X_test.shape[0]
random_predictions = np.random.randint(0, 2, size=num_samples)

# Gera um DataFrame no formato esperado da submissão
submission_df = pd.DataFrame({
    'ID': np.arange(1, num_samples + 1),
    'Prediction': random_predictions
})

# Salva o arquivo CSV no diretório atual
submission_df.to_csv('submission.csv', index=False)