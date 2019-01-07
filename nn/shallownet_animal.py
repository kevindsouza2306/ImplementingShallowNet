from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledataloader import SimpleDataLoader
from nn.conv.shallownet import ShallowNet

from keras.optimizers import SGD
from imutils import paths
import argparse
import matplotlib.pyplot as plt

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to dataset", required=True)

args = vars(ap.parse_args())

print("[INFO] loading images...")

imagePaths = list(paths.list_images(args["dataset"]))
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDataLoader(preprocessor=[sp, iap])

(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit(trainY)
testY = LabelBinarizer().fit(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)

model = ShallowNet.built(width=32, height=32, depth=3, classes=2)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training Network...")

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

print("[INFO] Evaluating the Network...")
prediction = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axix=1), prediction.argmax(axis=1)))
target_name = ["cat", "dog", "panda"]

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
