import scipy
from scipy import misc
import os
import matplotlib.pyplot as plt
import imageio

image_type = 'restaurant-with-sea-view'
path = '/Users/shiquan/PycharmProjects/Multi-modal Knowledge Base/images/restaurant/{}'.format(image_type)
new_path = '/Users/shiquan/PycharmProjects/Multi-modal Knowledge Base/images/restaurant/{} processed'.format(image_type)
if not os.path.exists(new_path):
    os.makedirs(new_path)

files = os.listdir(path)
count = 0

for file in files:
    print('{}'.format(file))
    if not os.path.isdir(file):
        try:
            image = imageio.imread(path + '/' + file)
        except:
            continue
        # plt.imshow(image)
        # plt.show()
        count += 1
        imageio.imsave(new_path + '/{}.png'.format(count), image)
