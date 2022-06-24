import numpy as np
import pandas as pd
from PIL import Image
try:
    import lzma
except ImportError:
    from backports import lzma
# first, we need to create the ncd function


def ncd(x,y):
    if type(x)=="numpy.ndarray":
        np.concatenate(x,y) # 
    else:
        x_y = x + y
    x_comp = lzma.compress(x)  # compress file 1
    y_comp = lzma.compress(y)  # compress file 2
    x_y_comp = lzma.compress(x_y)  # compress file concatenated
    ncd = (len(x_y_comp) - min(len(x_comp), len(y_comp))) / \
    max(len(x_comp), len(y_comp))
    return ncd

def ncd_vertical(x,y):
    x_y = np.concatenate((x,y),axis=1) # 
    x_comp = lzma.compress(x)  # compress file 1
    y_comp = lzma.compress(y)  # compress file 2
    x_y_comp = lzma.compress(x_y)  # compress file concatenated
    ncd = (len(x_y_comp) - min(len(x_comp), len(y_comp))) / \
    max(len(x_comp), len(y_comp))
    return ncd


def ncd_horizontal(x,y):
    x_y = np.concatenate((x,y),axis=0)
    x_comp = lzma.compress(x)  # compress file 1
    y_comp = lzma.compress(y)  # compress file 2
    x_y_comp = lzma.compress(x_y)  # compress file concatenated
    ncd = (len(x_y_comp) - min(len(x_comp), len(y_comp))) / \
    max(len(x_comp), len(y_comp))
    return ncd

#ncd versionns, tst with different versions

### test area

# load files from the dir

#test_array = np.load("data/01/dataset1/test/0/array.npy", allow_pickle=True)
#test_y = np.load("data/01/dataset1/test/0/y.npy", allow_pickle=True)

#train_array = np.load("data/01/dataset1/train/0/array.npy", allow_pickle=True)
#train_y = np.load("data/01/dataset1/train/0/y.npy", allow_pickle=True)

img1 = Image.open("data/01/dataset1/train/0/img_dump/image_0025.jpg")
img1= np.array(img1)
#print(type(test_array[1]))
x = open('data/01/dataset1/train/0/img_dump/image_0025.jpg', 'rb').read()
x3 = open('data/01/dataset1/train/0/img_dump/image_0010.jpg', 'rb').read()
#print(type(x))
img2 = Image.open("data/01/dataset1/train/0/img_dump/image_0001.jpg")
img2= np.array(img2)
y = open('data/01/dataset1/train/0/img_dump/image_0001.jpg', 'rb').read()
img3 = Image.open("data/01/dataset1/train/0/img_dump/image_0010.jpg")
img3= np.array(img3)
img4 = Image.open("data/01/dataset1/train/0/img_dump/image_0002.jpg")
img4= np.array(img4)
#train_df = pd.DataFrame(train_array)
#test = ncd(test_array[1],test_array[2])
test2 = ncd(x,y)
test3 = ncd(img1,img2)
test4 = ncd(img1,img3)
test42 = ncd(x,x3)
test5 = ncd(img1, img4)
print("Testing ncd with different file formats")
#print("npy")
#print(test)
print("jog: open") # we should use this one: but, is it normal that such a small difference?
print(test2)
print("image open")
print(test3)
print("image compare same image")
print(test4)
print("jpeg open: compare same image")
print(test42)
print("Image compare two hedgehogs")
print(test5)
#print(test_array)

# make a moving average of all images: with iamge open

# to start we take 5 images:

i1 = Image.open("data/01/dataset1/train/0/img_dump/image_0001.jpg")
i2 = Image.open("data/01/dataset1/train/0/img_dump/image_0002.jpg")
i3 = Image.open("data/01/dataset1/train/0/img_dump/image_0003.jpg")
i4 = Image.open("data/01/dataset1/train/0/img_dump/image_0004.jpg")
i5 = Image.open("data/01/dataset1/train/0/img_dump/image_0005.jpg")
i10 = Image.open("data/01/dataset1/train/0/img_dump/image_0010.jpg")
i25 = Image.open("data/01/dataset1/train/0/img_dump/image_0025.jpg")
#make moving average of the images

def moving_average(a,b,c,d,e):
    a= np.array(a,dtype=np.float)
    b= np.array(b,dtype=np.float)
    c= np.array(c,dtype=np.float)
    d= np.array(d,dtype=np.float)
    e= np.array(e,dtype=np.float)
    return (a+b+c+d+e)/5

final_img = moving_average(i1,i2,i3,i4,i5)

print("Distance taking into account a moving average of same type is:")
print(ncd(np.array(i10), final_img))


print("Distance taking into account a moving average of differnt type is:")
print(ncd(np.array(i25), final_img))


print("Distance taking into account a moving average of same type is:")
print(ncd(np.array(i10), final_img))


print("Distance taking into account a moving average of differnt type is:")
print(ncd(np.array(i25), final_img))


print("#################### same calculations but with vertical concat")

print("Distance taking into account a moving average of same type is:")
print(ncd_vertical(np.array(i10), final_img))


print("Distance taking into account a moving average of differnt type is:")
print(ncd_vertical(np.array(i25), final_img))


print("Distance taking into account a moving average of same type is:")
print(ncd_vertical(np.array(i10), final_img))


print("Distance taking into account a moving average of differnt type is:")
print(ncd_vertical(np.array(i25), final_img))


print("#################### same calculations but with horizontal concat")

print("Distance taking into account a moving average of same type is:")
print(ncd_horizontal(np.array(i10), final_img))


print("Distance taking into account a moving average of differnt type is:")
print(ncd_horizontal(np.array(i25), final_img))


print("Distance taking into account a moving average of same type is:")
print(ncd_horizontal(np.array(i10), final_img))


print("Distance taking into account a moving average of differnt type is:")
print(ncd_horizontal(np.array(i25), final_img))