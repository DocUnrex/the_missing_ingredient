import cv2
from skimage import io
url = "http://s0.geograph.org.uk/photos/40/57/405725_b17937da.jpg"
img = io.imread(url)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#clove detection
Clove_Cascades = cv2.CascadeClassifier("Resources/Clove_haarcascade.xml")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cloves = Clove_Cascades.detectMultiScale(imgGray, 1.1, 4)
for (x, y, z, w) in cloves:
        cv2.rectangle(img, (x, y), (x + z, y + w), (255, 0, 0), 2)
        cv2.putText(img, "Cloves", (x+ 5, y + w - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)

#cardamom detection
Cardamom_Cascades = cv2.CascadeClassifier("Resources/Cardamom_haarcascade.xml")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cardamom = Cardamom_Cascades.detectMultiScale(imgGray, 1.1, 4)
for (x, y, z, w) in cardamom:
        cv2.rectangle(img, (x, y), (x + z, y + w), (255, 0, 0), 2)
        cv2.putText(img, "Cardamom", (x+ 5, y + w - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)

#banana detection
Banana_Cascades = cv2.CascadeClassifier("Resources/Banana_haarcascade.xml")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
banana = Banana_Cascades.detectMultiScale(imgGray, 1.1, 4)
for (x, y, z, w) in banana:
        cv2.rectangle(img, (x, y), (x + z, y + w), (255, 0, 0), 2)
        cv2.putText(img, "Banana", (x+ 5, y + w - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)

#okra detection
Okra_Cascades = cv2.CascadeClassifier("Resources/Okra_haarcascade.xml")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
okra = Okra_Cascades.detectMultiScale(imgGray, 1.1, 4)
for (x, y, z, w) in okra:
        cv2.rectangle(img, (x, y), (x + z, y + w), (255, 0, 0), 2)
        cv2.putText(img, "Banana", (x+ 5, y + w - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)

cv2.imshow("Output",img)
cv2.waitKey(0)