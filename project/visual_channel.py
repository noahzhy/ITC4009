import cv2
import numpy as np
import matplotlib.pyplot as plt


channel_list = list()
channel_initials = list('BGR')
channel_labels = ['grayscale','Blue','Green','Red']
image = cv2.imread('project/images/lenna.jpg')

channel_list.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

for channel_index in range(3):
    channel = np.zeros(shape=image.shape, dtype=np.uint8)
    channel = image[:,:,channel_index]
    # cv2.imshow(f'{channel_initials[channel_index]}-RGB', channel)
    channel_list.append(channel)

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.title(channel_labels[i])
    plt.imshow(channel_list[i], cmap='gray')

plt.show()
