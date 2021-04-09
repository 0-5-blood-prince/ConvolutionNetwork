import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import cv2
import csv
labels = ["Amphibia","Animalia","Arachnida","Aves","Fungi","Insecta","Mammalia","Mollusca","Plantae","Reptilia"]
cwd = os.getcwd()
Dataset_Path = os.path.join(cwd , 'inaturalist_12K')
train_path = os.path.join(Dataset_Path , 'train')
test_path = os.path.join(Dataset_Path , 'val')
def loadImages():
    trainfiles = []
    trainclasses = []
    classcounts = [0 for i in range(10)]
    testfiles = []
    testclasses = []
    for subdir , dirs , files in os.walk(train_path):
        for file in files:
            if file.endswith(".jpg"):
                trainfiles.append(os.path.join(subdir,file))
                c = str(subdir).split(os.sep)[-1]
                for i in range(10):
                    if labels[i] == c:
                        trainclasses.append(i)
                        classcounts[i]+=1
                        break
    for subdir , dirs , files in os.walk(train_path):
        for file in files:
            if file.endswith(".jpg"):
                testfiles.append(os.path.join(subdir,file))
                c = str(subdir).split(os.sep)[-1]
                for i in range(10):
                    if labels[i] == c:
                        testclasses.append(i)
    return trainfiles , trainclasses , testfiles , testclasses , classcounts
def preprocess(data, height, width):
    dim = (width, height)
    resdata = []
    for i in range(len(data[:2000])):
        
        try:
            img = cv2.imread(data[i],cv2.IMREAD_UNCHANGED)
            res = cv2.resize(img, dim , interpolation=cv2.INTER_LINEAR)
            resdata.append(np.asarray(res))
        except Exception as e:
            print(data[i])
            print(str(e))
    return resdata

def dataset(width,height):
    trainfiles , trainclasses , testfiles , testclasses, classcounts = loadImages()
    train_data = preprocess(trainfiles, height,width)
    test_data = preprocess(testfiles, height,width)
    train_input = []
    val_input = []
    train_output = []
    val_output = []
    test_output = []
    i = 0
    cum = 0
    c = 0
    l = len(train_data)
    while(i<l):
        if i  <= cum + int((0.9)*classcounts[c]):
            train_input.append(train_data[i])
            train_output.append(trainclasses[i])
        else:
            val_input.append(train_data[i])
            val_output.append(trainclasses[i])
        i+=1
        if i== cum + classcounts[c]:
            cum += classcounts[c]
            c+=1
    test_input = []
    for i in range(len(test_data)):
        test_input.append(test_data[i])
        test_output.append(testclasses[i])
    train_output = np.eye(10)[train_output]
    val_output = np.eye(10)[val_output]
    test_output = np.eye(10)[test_output]
    
    L = len(train_data)
    return {
        'Xtrain' : np.array(train_input), 
        'Ytrain' : np.array(train_output),
        'Xval' : np.array(val_input),
        'Yval' : np.array(val_output),
        'Xtest' : np.array(test_input),
        'Ytest' :np.array(test_output)
    }
def flat(X):
    X_f = []
    for x in X:
        X_f.append(x.flatten())
    return X_f
def savedata(d): 
    with open('train_data.txt','w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(flat(d['Xtrain']))
# d = dataset(256,256)
# print(d['Xtrain'])
    
