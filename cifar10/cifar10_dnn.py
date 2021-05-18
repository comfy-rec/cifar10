from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Activation, MaxPooling2D, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
from keras import backend as K

# loss 그래프 그리기
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
actual_single = CLASSES[y_train]
plt.imshow(x_train[1000], interpolation='bicubic')
tmp = "label:" + str(actual_single[1000])
plt.title(tmp, fontsize=30)
plt.tight_layout()
plt.show()


NUM_CLASSES = 10

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

#모델 정의 부분(DNN)
#input_layer = Input(shape=(3072,), name='input_tensor')
input_layer = Input(shape=(32, 32, 3))
x = Flatten()(input_layer)
x = Dense(units=200, activation='relu')(x)
x = Dense(units=200, activation='relu')(x)
output_layer = Dense(units=10, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)
#################
#모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#################
#모델 훈련
history = model.fit(x_train, y_train, batch_size=16,
                    epochs=12, verbose=1, validation_split=0.2)
#################
#모델 평가
print('Test start')
score = model.evaluate(x_test, y_test, batch_size=16)
print('\nTest loss:', score[0])
print('Test acc:', score[1])

# 학습된 loss값과 acc를 보기위한 그래프
plot_loss(history)
plt.show()
plot_acc(history)
plt.show()
#################