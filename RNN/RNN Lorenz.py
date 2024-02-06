# Predict temporal series of Lorenz system
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# %% Lorenz Simulation functions

def lorenz(xyz, s=10, r=28, b=2.667) -> np.array:
    x,y,z=xyz
    x_dot = s*(y-x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    
    return np.array([x_dot, y_dot, z_dot])

def simulation(dt=0.01, num_steps=10000) -> None:
    
    xyzs=np.empty((num_steps+1, 3))
    xyzs[0]=(0., 1., 1.05)
    
    for i in range(num_steps):
        xyzs[i+1] = xyzs[i] + lorenz(xyzs[i])*dt
    
    return xyzs

# %% Simulation
xyzs = simulation()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(*xyzs.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title('Lorenz Attractor')

# %% Build the RNN

regressor = Sequential()

#first layer
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(60,1)))
regressor.add(Dropout(0.2))

#second layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#third layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#fourth layer
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))

# %% Train the RNN


# %% Predict