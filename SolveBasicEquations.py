import os as os
import cv2
import pandas as pd
import numpy as np
import pickle
import keras
from keras.models import Model
from keras.layers import * 
from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json



#Loading the training data
img_folder=r'./BIAI-Project/TrainingData'

def solve_equation(img):
    if img is not None:
        
        img=~img
        _,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        ctrs,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        w=int(28)
        h=int(28)
        train_data=[]
        rects=[]
        for c in cnt :
            x,y,w,h= cv2.boundingRect(c)
            rect=[x,y,w,h]
            rects.append(rect)
        
        bool_rect=[]
        for r in rects:
            l=[]
            for rec in rects:
                flag=0
                if rec!=r:
                    if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
                        flag=1
                    l.append(flag)
                if rec==r:
                    l.append(0)
            bool_rect.append(l)
        
        dump_rect=[]
        for i in range(0,len(cnt)):
            for j in range(0,len(cnt)):
                if bool_rect[i][j]==1:
                    area1=rects[i][2]*rects[i][3]
                    area2=rects[j][2]*rects[j][3]
                    if(area1==min(area1,area2)):
                        dump_rect.append(rects[i])
        
        final_rect=[i for i in rects if i not in dump_rect]
    
        for r in final_rect:
            x=r[0]
            y=r[1]
            w=r[2]
            h=r[3]
            im_crop =thresh[y:y+h+10,x:x+w+10]
            

            im_resize = cv2.resize(im_crop,(28,28))
            cv2.imshow("work",im_resize)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            im_resize=np.reshape(im_resize,(28,28,1))
            train_data.append(im_resize)
            
    s=''
    for i in range(len(train_data)):
        train_data[i]=np.array(train_data[i])
        train_data[i]=train_data[i].reshape(1,28,28,1)
    
        result= np.argmax(model.predict(train_data[i]), axis=-1)  #eception is thrown here-> Computed output size would be negative: -3 [input_size: 1, effective_filter_size: 5, stride: 1]

        if(result[0]==10):
            s=s+'-'
        if(result[0]==11):
            s=s+'+'
        if(result[0]==12):
            s=s+'*'
        if(result[0]==0):
            s=s+'0'
        if(result[0]==1):
            s=s+'1'
        if(result[0]==2):
            s=s+'2'
        if(result[0]==3):
            s=s+'3'
        if(result[0]==4):
            s=s+'4'
        if(result[0]==5):
            s=s+'5'
        if(result[0]==6):
            s=s+'6'
        if(result[0]==7):
            s=s+'7'
        if(result[0]==8):
            s=s+'8'
        if(result[0]==9):
            s=s+'9'
        
    print("Is the equation?: " + s)   
    print("Result: " + str(eval(s)))


#Load data from a folder
def load_images_from_folder(folder):
    train_data=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder ,filename),cv2.IMREAD_GRAYSCALE)
        img=~img
        if img is not None:
            _,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
            ctrs, _=cv2.findContours(thresh ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt=sorted(ctrs,key=lambda ctr:cv2.boundingRect(ctr)[0])
            w=int(28)
            h=int(28)
            maxi=0
            for c in cnt:
                x,y,w,h=cv2.boundingRect(c)
                maxi=max(w*h,maxi)
                if maxi==w*h:
                    x_max=x
                    y_max=y
                    w_max=w
                    h_max=h
                    im_crop= thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
                    im_resize = cv2.resize(im_crop,(28,28))
                    im_resize=np.reshape(im_resize,(784,1))
                    train_data.append(im_resize)                  
    return train_data

def createTestCSV():
    data=[]

    #assign '-'=10
    data=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\-')
    len(data)
    for i in range(0,len(data)):
        data[i]=np.append(data[i],['10'])
        
    print(len(data))

    #assign + = 11
    data11=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\+')

    for i in range(0,len(data11)):
        data11[i]=np.append(data11[i],['11'])
    data=np.concatenate((data,data11))
    print(len(data))
    
    #assign * = 12
    data12=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\times')

    for i in range(0,len(data12)):
        data12[i]=np.append(data12[i],['12'])
    data=np.concatenate((data,data12))
    print(len(data))

    data0=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\0')
    for i in range(0,len(data0)):
        data0[i]=np.append(data0[i],['0'])
    data=np.concatenate((data,data0))
    print(len(data))

    data1=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\1')

    for i in range(0,len(data1)):
        data1[i]=np.append(data1[i],['1'])
    data=np.concatenate((data,data1))
    print(len(data))

    data2=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\2')

    for i in range(0,len(data2)):
        data2[i]=np.append(data2[i],['2'])
    data=np.concatenate((data,data2))
    print(len(data))

    data3=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\3')

    for i in range(0,len(data3)):
        data3[i]=np.append(data3[i],['3'])
    data=np.concatenate((data,data3))
    print(len(data))

    data4=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\4')

    for i in range(0,len(data4)):
        data4[i]=np.append(data4[i],['4'])
    data=np.concatenate((data,data4))
    print(len(data))

    data5=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\5')

    for i in range(0,len(data5)):
        data5[i]=np.append(data5[i],['5'])
    data=np.concatenate((data,data5))
    print(len(data))

    data6=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\6')

    for i in range(0,len(data6)):
        data6[i]=np.append(data6[i],['6'])
    data=np.concatenate((data,data6))
    print(len(data))

    data7=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\7')

    for i in range(0,len(data7)):
        data7[i]=np.append(data7[i],['7'])
    data=np.concatenate((data,data7))
    print(len(data))

    data8=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\8')

    for i in range(0,len(data8)):
        data8[i]=np.append(data8[i],['8'])
    data=np.concatenate((data,data8))
    print(len(data))

    data9=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TestingData\9')

    for i in range(0,len(data9)):
        data9[i]=np.append(data9[i],['9'])
    data=np.concatenate((data,data9))
    print(len(data))

    df=pd.DataFrame(data,index=None)
    df.to_csv(r'.\BIAI-Project\test_final.csv',index=False)
    
    df_train=pd.read_csv(r".\BIAI-Project\train_final.csv",index_col=False)
    labels=df_train[['784']]

    df_train.drop(df_train.columns[[784]],axis=1,inplace=True)
    df_train.head()

    labels=np.array(labels)
    cat=to_categorical(labels,num_classes=13)
    print(cat[0])
    df_train.head()

    l=[]
    for i in range(9422):
        l.append(np.array(df_train[i:i+1]).reshape(28,28, 1))
    
    return l
    
def createCSV():
    data=[]

    #assign '-'=10
    data=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\-')
    len(data)
    for i in range(0,len(data)):
        data[i]=np.append(data[i],['10'])
        
    print(len(data))

    #assign + = 11
    data11=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\+')

    for i in range(0,len(data11)):
        data11[i]=np.append(data11[i],['11'])
    data=np.concatenate((data,data11))
    print(len(data))
    
    #assign * = 12
    data12=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\times')

    for i in range(0,len(data12)):
        data12[i]=np.append(data12[i],['12'])
    data=np.concatenate((data,data12))
    print(len(data))

    data0=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\0')
    for i in range(0,len(data0)):
        data0[i]=np.append(data0[i],['0'])
    data=np.concatenate((data,data0))
    print(len(data))

    data1=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\1')

    for i in range(0,len(data1)):
        data1[i]=np.append(data1[i],['1'])
    data=np.concatenate((data,data1))
    print(len(data))

    data2=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\2')

    for i in range(0,len(data2)):
        data2[i]=np.append(data2[i],['2'])
    data=np.concatenate((data,data2))
    print(len(data))

    data3=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\3')

    for i in range(0,len(data3)):
        data3[i]=np.append(data3[i],['3'])
    data=np.concatenate((data,data3))
    print(len(data))

    data4=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\4')

    for i in range(0,len(data4)):
        data4[i]=np.append(data4[i],['4'])
    data=np.concatenate((data,data4))
    print(len(data))

    data5=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\5')

    for i in range(0,len(data5)):
        data5[i]=np.append(data5[i],['5'])
    data=np.concatenate((data,data5))
    print(len(data))

    data6=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\6')

    for i in range(0,len(data6)):
        data6[i]=np.append(data6[i],['6'])
    data=np.concatenate((data,data6))
    print(len(data))

    data7=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\7')

    for i in range(0,len(data7)):
        data7[i]=np.append(data7[i],['7'])
    data=np.concatenate((data,data7))
    print(len(data))

    data8=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\8')

    for i in range(0,len(data8)):
        data8[i]=np.append(data8[i],['8'])
    data=np.concatenate((data,data8))
    print(len(data))

    data9=load_images_from_folder(r'C:\Users\meduz\Documents\BIAI-Project\TrainingData\9')

    for i in range(0,len(data9)):
        data9[i]=np.append(data9[i],['9'])
    data=np.concatenate((data,data9))
    print(len(data))

    df=pd.DataFrame(data,index=None)
    df.to_csv(r'.\BIAI-Project\train_final.csv',index=False)
    


def createModel():
    df_train=pd.read_csv(r".\BIAI-Project\train_final.csv",index_col=False)
    labels=df_train[['784']]

    df_train.drop(df_train.columns[[784]],axis=1,inplace=True)
    df_train.head()

    labels=np.array(labels)
    cat=to_categorical(labels,num_classes=13)
    df_train.head()

    l=[]
    for i in range(46785):
        l.append(np.array(df_train[i:i+1]).reshape(28,28, 1))

    #Create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(13, activation='softmax'))
    #Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(np.array(l), cat, epochs=10, batch_size=50,shuffle=True,verbose=1)
    #Save model
    model_json = model.to_json()
    with open(r".\BIAI-Project\model_final.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(r".\BIAI-Project\model_final.h5")


def testAccuracy():
    
    createTestCSV()
    df_train=pd.read_csv(r".\BIAI-Project\train_final.csv",index_col=False)
    labels=df_train[['784']]

    df_train.drop(df_train.columns[[784]],axis=1,inplace=True)
    df_train.head()

    labels=np.array(labels)
    cat=to_categorical(labels,num_classes=13)
    df_train.head()

    l=[]
    for i in range(46785):
        l.append(np.array(df_train[i:i+1]).reshape(28,28, 1))

    loss, accuracy = model.evaluate(np.array(l), cat) #evaluate the model
    print("Loss: " + str(loss)) #print the loss
    print("Accuracy: " + str(accuracy)) #print the accuracy


#createCSV()
createModel()

#Load the model


json_file = open(r'.\BIAI-Project\model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(r'.\BIAI-Project\model_final.h5')
#model = tf.keras.models.load_model('digits.model') #load the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
testAccuracy()

#Test the detection of made-by-hand digits
image_number = 1 #image number to be tested
      
while os.path.isfile(f"./BIAI-Project/digits/digit{image_number}.png"): 
    try:
        print(f"Testing image {image_number}...")
        img = cv2.imread(f"./BIAI-Project/digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)#read the image
        img = cv2.resize(img, (28,28)) #resize the image
        img = np.invert(np.array([img])) #invert the image
        prediction = model.predict(img) #predict the image
        print(f"The number is probably: {np.argmax(prediction)}") #print the prediction
    except:
        print("Error!")
    finally:
        image_number += 1 #increment the image number
   
#Solve the equation using bouding boxes   
equation_number = 1 #equation number to be tested

while os.path.isfile(f"./BIAI-Project/equations/equation{equation_number}.jpg"): 
    try:
        print(f"Testing equation {equation_number}...")
        img = cv2.imread(f"./BIAI-Project/equations/equation{equation_number}.jpg", cv2.IMREAD_GRAYSCALE)#read the image
        solve_equation(img)
    except:
        print("Error!")
    finally:
        equation_number += 1 #increment the image number



