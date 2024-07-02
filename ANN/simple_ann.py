#example code
#find the error

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main() -> None:
    n_samples = 1000
    n_features = 2
    n_classes = 5

    X = np.random.rand(n_samples, n_features) #1000x2
    y = np.random.randint(0, n_classes, size=n_samples) #gera entre zero e n_classes um número n_sample=1000
    print('y')
    print(y)
    y = np.eye(n_classes)[y] #converte para one hot encoding
    print('y2')
    print(y)

    print('X')
    print(X)
    return

    mean = np.mean(X, axis=0)
    std = np.std(X, axis = 0)
    X = (x-mean)/std

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_sate = 42)
    #model = ...
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    accuracy_score(y_test,y_pred)

if __name__ == "__main__":
    main()
