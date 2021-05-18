from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.utils import shuffle

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

iris = datasets.load_iris()
X = iris.data
Y = iris.target
print(X)
print(Y)

dataset_y = to_categorical(iris.target)
dataset_x = iris.data
dataset_x, dataset_y = shuffle(dataset_x, dataset_y)

X_train = dataset_x[:120]
x_test = dataset_x[120:]
y_train = dataset_y[:120]
y_test = dataset_y[120:]

# 네트워크 정의
# Sequential()
model = Sequential()
model.add(Dense(64, input_shape=(4,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 학습 시켜야죠
history = model.fit(X_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=1)

print('\n')
print('test start')
score = model.evaluate(x_test, y_test, verbose=1)
print('test loss:', score[0])
print('test acc:', score[1])

# 학습된 loss값과 acc를 보기위한 그래프
plot_loss(history)
plt.show()
plot_acc(history)
plt.show()
