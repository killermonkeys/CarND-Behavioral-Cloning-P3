import csv
import cv2
import numpy as np 
import random

#hyperparameters
sidecamcorrection = 0.18
epochs = 10
batch_size = 128
test_size = 0.2

#data path info
data_paths = [('track1-data/', 'track1-data/IMG/'),('track2-data/','track2-data/IMG/'),
              ('data-track1-2017-04-13-02/','data-track1-2017-04-13-02/IMG/'),
              ('data-track2-2017-04-13-03/', 'data-track2-2017-04-13-03/IMG/'),
              ('data-track2-2017-04-14-01/', 'data-track2-2017-04-14-01/IMG/'),
              ('data-track2-2017-04-14-02/', 'data-track2-2017-04-14-02/IMG/'),
              ('data-track1-2017-04-14-01/', 'data-track1-2017-04-14-01/IMG/'),
              ('data-track2-2017-04-14-03/', 'data-track2-2017-04-14-03/IMG/'),
              # this one is the "centerline drive", seems to have very bad effects
              #('data-track2-2017-04-15-01/', 'data-track2-2017-04-15-01/IMG/'),
              ('data-track1-2017-04-15-01/', 'data-track1-2017-04-15-01/IMG/')]

driving_log =[]
for (data_path, img_path) in data_paths:
    with open(data_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            measurement = float(line[3])
            center_file = img_path + line[0].split('/')[-1]
            left_file = img_path + line[1].split('/')[-1]
            right_file = img_path + line[2].split('/')[-1]
            #if random.choice([True, False]):
            driving_log.append([[center_file, 'normal'], measurement])
            driving_log.append([[left_file, 'normal'], measurement + sidecamcorrection])
            driving_log.append([[right_file, 'normal'], measurement - sidecamcorrection])
            #else:
            driving_log.append([[center_file, 'reverse'], -measurement])
            driving_log.append([[left_file, 'reverse'], -1 * (measurement + sidecamcorrection)])
            driving_log.append([[right_file, 'reverse'], -1 * (measurement - sidecamcorrection)])
        


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(driving_log, test_size=test_size)

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
                
                #hsvimg = normalizedHsv(image)
                #print("i 0- ", image[0].min(), "0+ ", image[0].max(), "1- ", image[1].min(), "1+ ", image[1].max(), "2- ", image[2].min(), "2+ ", image[2].max())
                #print("h 0- ", hsvimg[0].min(), "0+ ", hsvimg[0].max(), "1- ", hsvimg[1].min(), "1+ ", hsvimg[1].max(), "2- ", hsvimg[2].min(), "2+ ", hsvimg[2].max())
                images.append(image)

                measurement = float(batch_sample[1])
                measurements.append(measurement)

            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

model = Sequential()

from keras import backend as K
K.set_image_dim_ordering('tf')

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,15),(0,0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2),  activation='relu', padding='same'))
model.add(Conv2D(36, (5, 5), strides=(2, 2),  activation='relu', padding='same'))
model.add(Conv2D(48, (5, 5), strides=(2, 2),  activation='relu', padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# checkpoint
filepath="weights-{epoch:02d}-{val_loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min')
callbacks_list = [checkpoint]

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= 
            len(train_samples)/batch_size, validation_data=validation_generator, 
            validation_steps=len(validation_samples)/batch_size, epochs=epochs,
            callbacks=callbacks_list)

model.save('model.h5')

