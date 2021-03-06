import pandas as pd
import numpy as np
import glob
import model
from keras.utils import to_categorical

def get_training_data():
    count = 0
    train_array = np.zeros(shape=(32,32,32,4))
    flag_first_element = 1
    for voxel_file in glob.glob('./data/*.csv'):
        csv = pd.read_csv(voxel_file)
        (x_norm, y_norm, z_norm, occupany) = (csv.channel2.tolist(), csv.channel3.tolist(), csv.channel4.tolist(), csv.channel4.tolist())
        (x_cord, y_cord, z_cord) = (csv['ix'].tolist(), csv['iy'].tolist(), csv['iz'].tolist())
        _3d_x_norm =  _3d_y_norm = _3d_z_norm = _3d_occupancy = np.zeros(shape=(32,32,32))

        for i in range(len(x_cord)):
            _3d_x_norm[x_cord[i],y_cord[i],z_cord[i]] = x_norm[i]
            _3d_y_norm[x_cord[i],y_cord[i],z_cord[i]] = y_norm[i]
            _3d_z_norm[x_cord[i],y_cord[i],z_cord[i]] = z_norm[i]
            _3d_occupancy[x_cord[i],y_cord[i],z_cord[i]] = occupany[i]

        voxel = np.stack(( _3d_x_norm, _3d_y_norm, _3d_z_norm, _3d_occupancy),-1)

        if(flag_first_element):
            train_array = voxel
            flag_first_element = 0
        else:
            train_array = np.concatenate((train_array, voxel), -1)
        count = count + 1

    train_array = train_array.reshape(count, 32, 32, 32, 4)

    return train_array


def get_training_labels():
    csv = pd.read_csv('Path_to_labels_csv')
    labels = csv['labels']
    return labels.tolist()

def get_labels(y):
    labels = np.array(['No', 'Yes'])
    y = labels[y.argmax(1)]
    return y



if __name__ == '__main__':
    X_train = get_training_data()
    Y_train = [1,0]   #get_training_labels()
    Y_train = to_categorical(Y_train, 2)    #One hot encoding the data
    modell = model.create_model()
    modell.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    modell.fit(X_train, Y_train)
    y_test = modell.predict(X_train)
    y_test = get_labels(y_test)
    print(y_test)