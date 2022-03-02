import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
     ssl._create_default_https_context = ssl._create_unverified_context
     
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
nclasses = len(classes)


xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, random_state=9, train_size=7500, test_size=2500
)

XtrainScaled = xtrain / 255.0
XtestScaled = xtest / 255.0
clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(
    XtrainScaled, ytrain
)

ypredict = clf.predict(XtestScaled)
accuracy = accuracy_score(ytest, ypredict)


cap = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperLeft = (int(width / 2 - 56), int(height / 2 - 56))
        bottomRight = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)
        roi = gray[upperLeft[1] : bottomRight[1], upperLeft[0] : bottomRight[0]]
        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert("L")
        image_bwResized = image_bw.resize((28, 28), Image.ANTIALIAS)
        image_bwResizedInverted = PIL.ImageOps.invert(image_bwResized)
        pixelFilter = 20
        minPixel = np.percentile(image_bwResizedInverted, pixelFilter)
        image_bwResizedInvertedScaled = np.clip(
            image_bwResizedInverted - minPixel, 0, 255
        )
        maxPixel = np.max(image_bwResizedInverted)
        image_bwResizedInvertedScaled = (
            np.asarray(image_bwResizedInvertedScaled) / maxPixel
        )
        testSample = np.array(image_bwResizedInvertedScaled).reshape(1, 784)
        testpredict = clf.predict(testSample)
        print("Predicted class is: ", testpredict)

        cv2.imshow("frame", gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
