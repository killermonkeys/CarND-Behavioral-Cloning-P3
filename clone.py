import csv
import cv2
import numpy as np 

#hyperparameters
sidecamcorrection = 0.4
epochs = 8
test_size = 0.3

#data path info
data_path = 'alldata/'
img_path = data_path + 'IMG/'

driving_log =[]
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        measurement = float(line[3])
        center_file = img_path + line[0].split('/')[-1]
        driving_log.append([[center_file, 'normal'], measurement])
        driving_log.append([[center_file, 'reverse'], -measurement])
        left_file = img_path + line[1].split('/')[-1]
        driving_log.append([[left_file, 'normal'], measurement + sidecamcorrection])
        driving_log.append([[left_file, 'reverse'], -1 * (measurement + sidecamcorrection)])
        right_file = img_path + line[2].split('/')[-1]
        driving_log.append([[right_file, 'normal'], measurement - sidecamcorrection])
        driving_log.append([[right_file, 'reverse'], -1 * (measurement - sidecamcorrection)])
        


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(driving_log, test_size=0.3)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

def normalizedHsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
    #H is 0 - 180
    #hsv_img[0] = hsv_img[0] /180.0 - 0.5
    # rest are 0-255
    #hsv_img[1] = hsv_img[1] / 255.0 - 0.5
    #hsv_img[2] = hsv_img[2] / 255.0 - 0.5
    return hsv_img

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0][0])
                #flip image if it's a reverse image (measurement already flipped)
                if (batch_sample[0][1] == 'reverse'):
                    image = cv2.flip(image, 1)

                #contrast equalize image
                #lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                #lab_image[0] = clahe.apply(lab_image[0])
                #image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
                
                hsvimg = normalizedHsv(image)
                #print("i 0- ", image[0].min(), "0+ ", image[0].max(), "1- ", image[1].min(), "1+ ", image[1].max(), "2- ", image[2].min(), "2+ ", image[2].max())
                #print("h 0- ", hsvimg[0].min(), "0+ ", hsvimg[0].max(), "1- ", hsvimg[1].min(), "1+ ", hsvimg[1].max(), "2- ", hsvimg[2].min(), "2+ ", hsvimg[2].max())
                images.append(hsvimg)

                measurement = float(batch_sample[1])
                measurements.append(measurement)

            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((30,10),(0,0))))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=epochs)

model.save('model.h5')

