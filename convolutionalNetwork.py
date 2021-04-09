from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

class ConvNet:
    def __init__(self, numFilters, filterSize, filterConf = 'same'):
        return

    def initModel(self, numFilters, filterSize, filterConf = 'same'):
        model = Sequential()
        model.add(Conv2D(numFilters, (filterSize, filterSize), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))        
        
        for l in range(4):
            model.add(Conv2D(numFilters, (filterSize, filterSize)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(FLatten())
        #model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        