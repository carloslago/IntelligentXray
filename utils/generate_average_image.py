import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import progressbar as pb



# 191229 frontales + 32419 laterales

#initialize widgets
# widgets = ['Time for loop of 225000 iterations: ', pb.Percentage(), ' ',
#             pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
#
# timer = pb.ProgressBar(widgets=widgets, maxval=250000).start()

img_path = os.path.join('CheXpert-v1.0-small')

cont = 0
cont_front = 0
cont_lat = 0
img_frontales = []
img_laterales = []
i=0
for subdir, dirs, files in os.walk(img_path):
    for file in files:
        current = subdir + '/' + file
        try:
            # if file.split('_')[1] == "frontal.jpg":
                # cont_front += 1

            if file.split('_')[1] == "lateral.jpg":
                # if cont >=50000:
                im = Image.open(current, mode='r')
                im = im.resize((390, 320))
                img_frontales.append(im)
                # cont_lat += 1
            # else:
            #     pass
        except:
            pass
    cont+=1
    # timer.update(cont)
    if cont%10000==0: print(cont)
    # if cont>=40000: break



# timer.finish()
# print(cont_lat)
# print(cont_front)


ims = np.array([np.array(im) for im in img_frontales])
imave_media = np.average(ims,axis=0)
imave_median = np.median(ims,axis=0)
result1 = Image.fromarray(imave_media.astype('uint8'))
result2 = Image.fromarray(imave_median.astype('uint8'))
result1.save('mean_lateral.jpg')
result2.save('median_lateral.jpg')
