from os import listdir
from random import shuffle

path = 'images'
list_str = ""
list_images = []
for line in listdir(path):
    list_images.append(line + " %d" % (int(line.split("_")[1])))

shuffle(list_images)


train_size = int(len(list_images)*0.8)
test_size = len(list_images) - train_size

open('listnames_test.txt', 'w').write("\n".join(list_images[0:test_size]))
open('listnames_train.txt', 'w').write("\n".join(list_images[test_size:]))