from keras.models import load_model
import numpy as np
import cv2
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledataloader import SimpleDataLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to dataset", required=True)
ap.add_argument("-m", "--model", help="path to model", required=True)

args = vars(ap.parse_args())

classlabels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("[INFO] sampling imags")

imagePaths = np.array(list(paths.list_images(args["dataset"])))

idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDataLoader(preprocessor=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

print("[INFO] predicting....")

preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, imagePaths) in enumerate(imagePaths):
    image = cv2.imread(imagePaths)
    cv2.putText(image, "Label: {}".format(classlabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                2)
    cv2.imshow("image",image)
    cv2.waitKey(0)
