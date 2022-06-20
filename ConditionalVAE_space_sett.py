import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from tensorflow.python.keras.layers import Input, Dense, Concatenate
from tensorflow.python.keras.models import Model


# create artificial data space
x1 = np.linspace(-2.2, 2.2, 1000)
fx1 = np.sin(x1)
dots1 = np.vstack([x1, fx1]).T

t = np.linspace(0, 2 * np.pi, num=1000)
dots2 = 0.5 * np.array([np.sin(t), np.cos(t)]).T + np.array([1.5, -0.5])[None, :]

dots = np.vstack([dots1, dots2])
noise = 0.06 * np.random.randn(*dots.shape)

labels = np.array([0]*1000 + [1]*1000)
noised = dots + noise


# Visualisation
colors = ['b'] * 1000 + ['g'] * 1000
'''plt.figure(figsize=(15, 15))
plt.xlim([-2.5, 2.5])
plt.ylim([-1.5, 1.5])
plt.scatter(noised[:, 0], noised[:, 1], c=colors)
plt.plot(dots1[:, 0], dots1[:, 1], color='red', linewidth=4)
plt.plot(dots2[:, 0], dots2[:, 1], color='yellow', linewidth=4)
plt.grid(False)'''


# simple Model and its training
def dense_ae():
    inp_dots = Input((2, ))
    inp_lbls = Input((1, ))
    inp = Concatenate()([inp_dots, inp_lbls])
    x = Dense(64, activation='relu')(inp)
    x = Dense(64, activation='relu')(x)
    code = Dense(1, activation='linear')(x)

    fcode = Concatenate()([code, inp_lbls])
    x = Dense(64, activation='relu')(fcode)
    x = Dense(64, activation='relu')(x)
    out = Dense(2, activation='linear')(x)

    ae = Model([inp_dots, inp_lbls], out)
    return ae


dae = dense_ae()
dae.compile(optimizer='adam', loss='mse')
dae.fit([noised, labels], noised, epochs=50, batch_size=40, verbose=2)

# Result
predicted = dae.predict([noised, labels])

# So, if we give as input only data without labels (what category this data is)
# than we will get not so good results: we couldn't clear define and separate different categories

# visualize predicted
plt.figure(figsize=(15, 9))
plt.xlim([-2.5, 2.5])
plt.ylim([-1.5, 1.5])
plt.scatter(noised[:, 0], noised[:, 1], c=colors)
plt.plot(dots1[:, 0], dots1[:, 1],  color="red",    linewidth=4)
plt.plot(dots2[:, 0], dots2[:, 1],  color="yellow", linewidth=4)
plt.scatter(predicted[:, 0], predicted[:, 1], c='gray', s=50)
plt.grid(False)
plt.show()