import cv2
from matplotlib import pyplot as plt
img1 = cv2.imread('image1.png',0)
img2 = cv2.imread('image2.png',0)
edges1 = cv2.Canny(img1,500,10)
edges2 = cv2.Canny(img2,100,200)
plt.subplot(121),plt.imshow(edges1,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges2,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()