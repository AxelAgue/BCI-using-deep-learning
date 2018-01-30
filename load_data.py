import numpy as np
import h5py
import matplotlib.pyplot as plt
import sklearn


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

f3 = h5py.File('/home/matt/PycharmProjects/A-Generalizable-BCI-using-Machine-Learning-for-Feature-Discovery/data/EEG_ParticipantC_30_09_2013.mat','r')
data3 = f3.get("data_epochs_C")
label3 = f3.get("data_key_C")
data3 = np.array(data3)
label3 = np.array(label3)

f4 = h5py.File('/home/matt/PycharmProjects/A-Generalizable-BCI-using-Machine-Learning-for-Feature-Discovery/data/EEG_ParticipantD_01_10_2013.mat','r')
data4 = f4.get("data_epochs_D")
label4 = f4.get("data_key_D")
data4 = np.array(data4)
label4 = np.array(label4)

f5 = h5py.File('/home/matt/PycharmProjects/A-Generalizable-BCI-using-Machine-Learning-for-Feature-Discovery/data/EEG_ParticipantE_31_10_2013.mat','r')
data5 = f5.get("data_epochs_E")
label5 = f5.get("data_key_E")
data5 = np.array(data5)
label5 = np.array(label5)


"""
data = []
for i in range(len(data1)):
    data.append(data1[i])
for j in range(len(data2)):
    data.append(data2[j])
for h in range(len(data3)):
    data.append(data3[h])
for j in range(len(data4)):
    data.append(data4[j])
for h in range(len(data5)):
    data.append(data5[h])


label = []
for i in range(len(label1)):
    label.append(label1[i])
for j in range(len(label2)):
    label.append(label2[j])
for h in range(len(label3)):
    label.append(label3[h])
for i in range(len(label4)):
    label.append(label4[i])
for i in range(len(label5)):
    label.append(label5[i])


"""

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



""""

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

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
new_data = mms.fit_transform(data)




from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
new_label = ohe.fit_transform(label)






"""




"""


t = np.arange(0, 6200, 1)
fig, axs = plt.subplots(3, 1)

axs[0].plot(t, data[7])
axs[0].set_xlim(5050, 5350)
axs[0].set_xlabel('time')
axs[0].set_ylabel('s1')
axs[0].title.set_text(str(label[7]))
axs[0].grid(True)

axs[1].plot(t, data[99])
axs[1].set_xlim(5050, 5350)
axs[1].set_xlabel('time')
axs[1].set_ylabel('s1')
axs[1].title.set_text(str(label[99]))
axs[1].grid(True)

axs[2].plot(t, data[15])
axs[2].set_xlim(5050, 5350)
axs[2].set_xlabel('time')
axs[2].set_ylabel('s1')
axs[2].title.set_text(str(label[15]))
axs[2].grid(True)


fig.tight_layout()
plt.show()
"""