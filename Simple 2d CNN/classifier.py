train_data_dir = 'train'
validation_data_dir = 'test'
nb_train_samples = 30000
nb_validation_samples = 9000

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
img_width, img_height = 120, 96

epochs = 2
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
def train():
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('hockey_wights.h5')

model.load_weights('hockey_wights.h5')
import numpy as np
from keras.preprocessing import image

def predict_helper(path):
  print train_generator.class_indices
  testimg = image.load_img(path, target_size = (img_width,img_height))
  testimg = image.img_to_array(testimg)
  testimg = np.expand_dims(testimg, axis = 0)
  return model.predict_classes(testimg)

def predictvideo(path, name):
    #path should end with /
    video = cv2.VideoCapture(path+name)

    frame = video.read()
    frames = 0
    while frame[0]:
        # get frame by frame
        frames += 1
        frame = video.read()
        cv2.imwrite(name + '_frame{}'.format(frames) + '.jpg',frame[1])

    print 'Created {} frames from {}'.format(frames, name)
    video.release()
     
    return
