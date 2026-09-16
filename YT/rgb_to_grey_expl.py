import cv2
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread("ton_image.jpg"), cv2.COLOR_BGR2RGB)
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
gray = 0.2989*R + 0.5870*G + 0.1140*B

fig, axs = plt.subplots(1, 4, figsize=(12,4))
axs[0].imshow(R, cmap='gray'); axs[0].set_title('Canal R')
axs[1].imshow(G, cmap='gray'); axs[1].set_title('Canal G')
axs[2].imshow(B, cmap='gray'); axs[2].set_title('Canal B')
axs[3].imshow(gray, cmap='gray'); axs[3].set_title('Niveaux de gris')
for ax in axs: ax.axis('off')
plt.show()
