#Skin Segmentation Data Set from https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
#face.png image from http://graphics.cs.msu.ru/ru/node/899

import numpy as np
import cv2
import glob


from sklearn import svm
from sklearn import tree
from sklearn.cross_validation import train_test_split

def ReadData():
    #Data in format [B G R Label] from
    data = np.genfromtxt('./data/Skin_NonSkin.txt', dtype=np.int32)

    labels= data[:,3]
    data= data[:,0:3]

    return data, labels

def BGR2HSV(bgr):
    bgr= np.reshape(bgr,(bgr.shape[0],1,3))
    hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv= np.reshape(hsv,(hsv.shape[0],3))

    return hsv

def TrainTree(data, labels, flUseHSVColorspace):
    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

    print(trainData.shape)
    print(trainLabels.shape)
    print(testData.shape)
    print(testLabels.shape)

    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = svm.SVC()
    clf = clf.fit(trainData, trainLabels)
    # print(clf.feature_importances_)
    print(clf.score(testData, testLabels))

    return clf

def ApplyToImage(path,clf, flUseHSVColorspace,counter):

    img= cv2.imread(path)
    print(img.shape)
    data= np.reshape(img,(img.shape[0]*img.shape[1],3))
    print(data.shape)

    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    predictedLabels= clf.predict(data)

    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))

    filename = './results/' + str(counter).zfill(3)
    if (flUseHSVColorspace):
        filename = filename + '_result_HSV.png'
        cv2.imwrite('./results/result_HSV.png',((-(imgLabels-1)+1)*255))# from [1 2] to [0 255]
    else:
        filename = filename + '_result_RGB.png'
    
    cv2.imwrite(filename,((-(imgLabels-1)+1)*255))


#---------------------------------------------

data, labels= ReadData()
clf_hsv = TrainTree(data, labels, True)
clf_rgb = TrainTree(data, labels, False)
# ApplyToImage("face.png", clf_hsv, True, 0)
# ApplyToImage("face.png", clf_rgb, False, 0)
imagesPath = glob.glob('./input/*.jpg')
counter = 0
for imagePath in imagesPath:
    ApplyToImage(imagePath,clf_hsv, True, counter)
    ApplyToImage(imagePath,clf_rgb, False, counter)
    counter = counter +1