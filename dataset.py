import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
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
    for i in range(len(data[:3])):
        
        try:
            img = cv2.imread(data[i],cv2.IMREAD_UNCHANGED)
            res = cv2.resize(img, dim , interpolation=cv2.INTER_LINEAR)
            resdata.append(res)
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
            train_input.append(np.asarray(train_data[i]))
            train_output.append(trainclasses[i])
        else:
            val_input.append(np.asarray(train_data[i]))
            val_output.append(trainclasses[i])
        i+=1
        if i== cum + classcounts[c]:
            cum += classcounts[c]
            c+=1
    test_input = []
    for i in range(len(test_data)):
        test_input.append(np.asarray(test_data[i]))
        test_output.append(testclasses[i])
    train_output = np.eye(10)[train_output]
    val_output = np.eye(10)[val_output]
    train_output = np.eye(10)[test_output]
    
    L = len(train_data)
    return {
        'Xtrain' : train_input, 
        'Ytrain' : train_output,
        'Xval' : val_input,
        'Yval' : val_output,
        'Xtest' : test_input,
        'Ytest' :test_output
    }
