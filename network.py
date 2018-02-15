import numpy as np
import h5py
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


f1 = h5py.File('/home/matt/PycharmProjects/A-Generalizable-BCI-using-Machine-Learning-for-Feature-Discovery/data/EEG_ParticipantA_28_09_2013.mat','r')
data1 = f1.get("data_epochs_A")
label1 = f1.get("data_key_A")
data1 = np.array(data1)
label1 = np.array(label1)

f2 = h5py.File('/home/matt/PycharmProjects/A-Generalizable-BCI-using-Machine-Learning-for-Feature-Discovery/data/EEG_ParticipantB_30_09_2013.mat','r')
data2 = f2.get("data_epochs_B")
label2 = f2.get("data_key_B")
data2 = np.array(data2)
label2 = np.array(label2)


for i in range(len(data1)):
    new_data1 = data1[i]
    for j in range(len(new_data1)):
        if new_data1[j] > 0:
            new_data1[j] = new_data1[j] / 12969.0559545

        else:
            new_data1[j] = new_data1[j] / 14559.7649705

for i in range(len(data2)):
    new_data2 = data2[i]
    for j in range(len(new_data2)):
        if new_data1[j] > 0:
            new_data1[j] = new_data1[j] / 6066.65883057

        else:
            new_data1[j] = new_data1[j] / 2006.3681663




data = []
for i in range(len(data1)):
    data.append(data1[i])
for j in range(len(data2)):
    data.append(data2[j])

label = []
for i in range(len(label1)):
    label.append(label1[i])
for j in range(len(label2)):
    label.append(label2[j])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
new_label = ohe.fit_transform(label)




X_train, X_test, y_train, y_test = train_test_split(data, new_label, test_size=0.10, random_state=0)

train_data = X_train
train_data_labels = y_train


test_data = X_test
test_data_labels = y_test
k = 30


class Network(object):
    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def mini_batch(self, mini_batch_size, epochs, train_data_labels, train_data, test_data=None,
                   test_data_labels=None, cost_func=None):

        # eleccion del mini batch

        # np.random.seed(5)
        cost = []
        eta = 5

        for j in range(epochs):
            """
            h = np.random.randint(0, len(train_data_labels) - mini_batch_size)
            mini_batch = train_data[h:h + mini_batch_size]
            mini_batch_labels = train_data_labels[h:h + mini_batch_size]
            """

            mini_batch = train_data


            mini_batch_labels = train_data_labels

            self.GD_actualizacion_parametros(mini_batch, mini_batch_labels, eta, mini_batch_size)


            if test_data is not None:

                if cost_func == "cuadratic cost":
                    prediccion_correcta, mse = self.cc_evaluate(test_data, test_data_labels)
                    error = 1 - (prediccion_correcta / 2745)
                    cost.append(error)
                    print("Epoch {0}. Predicciones Correctas: {1} de 2745. Error con Cuadratic Cost: {2}".format(j,
                                                                                                                  prediccion_correcta,
                                                                                                                  error))

                if cost_func == "cross entropy":
                    prediccion_correcta, ce = self.ce_evaluate(test_data, test_data_labels)
                    error = 1 - (prediccion_correcta / 2745)
                    cost.append(error)
                    print("Epoch {0}. Predicciones Correctas: {1} de 2745. Error con Cross Entropy: {2}".format(j,
                                                                                                                 prediccion_correcta,
                                                                                                                 error))

        self.plot_cost(epochs, cost)

    def GD_actualizacion_parametros(self, mini_batch, mini_batch_labels, eta, mini_batch_size):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for i in range(mini_batch_size):
            s = mini_batch[i]
            y = mini_batch_labels[i]

            delta_nabla_b, delta_nabla_w = self.backprop(s, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]




        # print(nabla_w[1])

    def backprop(self, s, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs = []
        activations = [s]

        a = np.dot(self.weights[0], s) + self.biases[0].reshape(1, k)
        zs.append(a)
        z = act_func(a, "sigmoid")
        activations.append(z)
        b = np.dot(self.weights[1], z.reshape(k, 1))
        zs.append(b)
        z1 = act_func(b, "softmax")
        activations.append(z1)

        delta = (activations[-1] - y.reshape(3,1))  # * act_func_prime(zs[-1], "sigmoid")

        nabla_b[-1] = delta
        nabla_w[-1] = np.multiply(delta, activations[-2].reshape(1, k))

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = act_func_prime(z, "sigmoid")
            delta = np.dot(self.weights[-l + 1].reshape(k, 3), delta) * sp.reshape(k, 1)

            nabla_b[-l] = delta
            nabla_w[-l] = np.multiply(delta, activations[-l - 1])

        return nabla_b, nabla_w

    def feedforward(self, z):

        # for i in range(self.num_layers - 1):
        a = act_func((np.dot(self.weights[0], z) + self.biases[0].reshape(1, k)), "sigmoid")
        z = act_func(np.dot(self.weights[1], a.reshape(k, 1)) + self.biases[1], "softmax")

        return z

    def cc_evaluate(self, test_data, test_data_labels):

        c1 = [self.feedforward(x) for x in test_data]
        c2 = test_data_labels

        prediccion_correcta = 0

        for i in range(len(test_data)):

            mse = ((c1[i] - c2[i]) ** 2).mean()

            mse += mse

            if np.argmax(c1[i]) == np.argmax(c2[i]):
                prediccion_correcta += 1

        return prediccion_correcta, mse

    def plot_cost(self, epochs, cost):

        plt.figure()
        plt.plot(range(epochs), cost)
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.show()

    def ce_evaluate(self, test_data, test_data_labels):

        ce = 0
        c1 = [self.feedforward(x) for x in test_data]
        c2 = test_data_labels

        prediccion_correcta = 0

        for i in range(len(test_data)):

            ce = np.sum(np.nan_to_num(-c2[i] * np.log(c1[i]) - (1 - c2[i]) * np.log(1 - c1[i])))

            """
            c1[i] = np.clip(c1[i], epsilon, 1. - epsilon)
            ce = - np.mean(np.log(c1[i]) * c2[i])

            """
            ce += ce

            if np.argmax(c1[i]) == np.argmax(c2[i]):
                prediccion_correcta += 1

        return prediccion_correcta, ce


def act_func(z, type):
    if type == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))

    if type == "relu":
        return z * (z > 0)

    if type == "lineal":
        return z

    if type == "softmax":
        return np.exp(z) / np.sum(np.exp(z), axis=0)


def act_func_prime(z, type):
    if type == "sigmoid":
        return act_func(z, "sigmoid") * (1 - act_func(z, "sigmoid"))

    if type == "relu":
        return 1. * (z > 0)

    if type == "lineal":
        return 1

    if type == "softmax":
        out = np.zeros(np.shape(z))
        out[:, :-1] = -z[:, :-1] * z[:, :-1]
        out[:, -1] = z[:, -1] * (1 - z[:, -1])
        return out


net = Network([6200, 30, 3])
net.mini_batch(100, 1000, train_data_labels, train_data, test_data=test_data, test_data_labels=test_data_labels,
               cost_func="cuadratic cost")

