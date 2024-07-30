from system import MassSpringDamper
import numpy as np
from tensorflow import keras
import tensorflow as tf

def physics(model, x, m, c, k, f):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        u = model(x)
    u_t = tape.gradient(u, x)
    u_tt = tape.gradient(u_t, x)
    f_pred = m * u_tt + c * u_t + k * u
    return f_pred - f

def loss(model, x_train, t_train, x_test, t_test, m, c, k, f):
    with tf.GradientTape() as tape:
        u_pred = model(x_train)
        f_pred = physics(model, x_train, m, c, k, f)
        loss_f = tf.reduce_mean(tf.square(f_pred))

        u_pred_test = model(x_test)
        loss_data = tf.reduce_mean(tf.square(u_pred_test - t_test))

        total_loss = loss_f + loss_data

    grads = tape.gradient(total_loss, model.trainable_variables)
    return total_loss, grads

def get_network():
    X = np.random.rand(n, m)
    y = np.random.randint(0, n_classes, size=n)
    input_layer = keras.layers.Input(shape=X.shape[-1])
    H = keras.layers.Dense(units=8, activation='tanh', kernel_initializer=initializer)(inputs)
    H = keras.layers.Dense(units=16, activation='tanh', kernel_initializer=initializer)(H)
    H = keras.layers.Dense(units=32, activation='tanh', kernel_initializer=initializer)(H)
    H = keras.layers.Dense(units=64, activation='tanh', kernel_initializer=initializer)(H)
    outputs = keras.layers.Dense(units=n_classes, activation='softmax', kernel_initializer=initializer)(H)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    return model

def main():
    model = get_network()
    num_epochs=10
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            loss_value, grads = loss(model, x_train, t_train, x_test, t_test, m, c, k, f)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Epoch: {epoch}, Loss: {loss_value}")

def main2() -> None:
    massSpringDamper = MassSpringDamper(1,2,0.5)
    massSpringDamper.solve_system(y0=[0,1],t0=0,tf=30,dt=0.1)
    sol = massSpringDamper.get_solution()
    print(sol)
    massSpringDamper.show_plot()

if __name__ == '__main__':
    main()