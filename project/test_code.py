# import cv2
# import numpy as np

# channel_initials = list('BGR')

# image = cv2.imread('project/lenna.jpg')

# for channel_index in range(3):
#     channel = np.zeros(shape=image.shape, dtype=np.uint8)
#     channel = image[:,:,channel_index]
#     cv2.imshow(f'{channel_initials[channel_index]}-RGB', channel)

# cv2.waitKey(0)
channel = "r"
idx = list('BGR').index(channel.upper())
print(idx)

