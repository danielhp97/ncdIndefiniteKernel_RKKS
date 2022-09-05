from pyexpat import model
from re import L
import sys
sys.path.insert(1, './src/00/')
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import paq as pq
from io import BytesIO
import time
from PIL import Image
from PIL.ImageFilter import MedianFilter, BLUR 
from learning_algorithms import  SquareHingeKreinSVM, kernel_matrix
from ncd import  Ncd
try:
    import lzma
except ImportError:
    from backports import lzma
     
if __name__ == "__main__":
    #i = sys.argv[1]
    #i = str(i)
    # compute the kernel
    #open parameter file
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    params=params['Parameters']
    dataset_number = params['dataset']
    ncd_type = params['ncd_baseline']['ncd_type']
    
    com_images = np.load("data/dataset1/kernel/compressedImagesSize.npy", allow_pickle=True)
    kernel_matrix = np.load("data/dataset1/kernel/kernelmatrix.npy")
    
    ax = sns.heatmap(kernel_matrix)
    sns.color_palette("rocket", as_cmap=True)
    plt.savefig('kernelInterleaved.png', bbox_inches='tight')
    plt.show()

    print("Kernel Matrix:")
    print(pd.DataFrame(kernel_matrix))
    original_img1 = com_images[2,:,:,:]
    original_img2 = com_images[34,:,:,:]
    calc = Ncd()
    print("________ compressed images test_________")
    #length image 1
    img1 = com_images[2,:,:,:]
    img1 = np.asarray(img1)
    img1 = Image.fromarray(img1)
    byteimg1 = BytesIO()
    img1.save(byteimg1, 'jpeg')
    lenimg1 = byteimg1.tell()


    #length image 2
    img2 = com_images[34,:,:,:]
    img2 = np.asarray(img2)
    img2 = Image.fromarray(img2)
    byteimg2 = BytesIO()
    img2.save(byteimg2, 'jpeg')
    lenimg2 = byteimg2.tell()


    print("Length of Image 1: {}".format(lenimg1))
    print("Length of Image 2: {}".format(lenimg2))

    print("________ ncd test_________")
    test = calc.ncd_jpeg2000(img1,img2,lenimg1,lenimg2)
    print("Ncd with Horizontal Concatenation: {}".format(test))
    test = calc.ncd_jpeg2000vertical(img1,img2,lenimg1,lenimg2)
    print("Ncd with Vertical Concatenation: {}".format(test))
    

    print("________Same image tests_____________")
    #test = calc.ncd_jpeg2000(img1,img1,lenimg1,lenimg1)
    #print("Ncd with the same image: {}".format(test))
    
    img1 = com_images[2,:,:,:]
    img1 = np.asarray(img1)
    img1 = Image.fromarray(img1)
    img1.save('temp/byteimg1.jpeg')
    lenimg1 = os.stat('temp/byteimg1.jpeg').st_size
    print("Length of the 1 image used: {}".format(lenimg1))

    img3 = com_images[2,:,:,:]
    img3 = np.asarray(img3)
    img3 = Image.fromarray(img3)
    img3.save('temp/byteimg3.jpeg')
    lenimg3 = os.stat('temp/byteimg3.jpeg').st_size
    print("Length of the 2 image used: {}".format(lenimg1))


    x_y_img1 = np.concatenate((original_img1,original_img1), axis = 0)
    x_y_img1 = Image.fromarray(x_y_img1) # save as temp and compare
    x_y_img1.save('temp/bytexy1.jpeg')
    lenx_y_img1 = os.stat('temp/bytexy1.jpeg').st_size
    print("Length of the concat of the 1 image: {}".format(lenx_y_img1))
    #x_y_img2 = np.concatenate((original_img1,original_img1), axis = 0)
    #x_y_img2 = Image.fromarray(x_y_img2)
    #bytes_img2 = BytesIO()
    #x_y_img1.save(bytes_img2,'jpeg')
    #lenx_y_img2 = bytes_img2.tell()    
    #print("Length of the concat of the 2 image: {}".format(lenx_y_img2))
    # 16797 / 7734
    ncd = (lenx_y_img1 - min(lenimg1, lenimg3)) / max(lenimg1, lenimg3)
    print("Ncd Result: {}".format(ncd))

    print("________Paq Tests_____________")

    def image_to_byte_array(image: Image) -> bytes:
        #image = Image.fromarray(image)
        imgByteArr = BytesIO()
        print(image.format)
        image.save(imgByteArr, format="JPEG")
        imgByteArr = imgByteArr.getvalue()
        return imgByteArr

    concatx_y = Image.new("RGB", (600, 240), "white")
    x = original_img1
    x = Image.fromarray(x)
    x.save("temp/x_testimg.jpeg")
    y = original_img2
    y = Image.fromarray(y)
    y.save("temp/y_testimg.jpeg")
    x_original = Image.fromarray(original_img1)
    y_original = Image.fromarray(original_img1)
    concatx_y.paste(x, (0, 0))
    concatx_y.paste(y, (300,0))
    byteconcat = BytesIO()
    concatx_y.save(byteconcat, 'jpeg')
    temp_concat = concatx_y.save("temp/kernel_testimg.jpeg")
    temp_concat = Image.open("temp/kernel_testimg.jpeg")
    lenx_y = len(pq.compress(image_to_byte_array(temp_concat)))
    print("this one is good")
    len_x_comp = len(pq.compress(image_to_byte_array(x_original)))
    print("this one is good 2")
    len_y_comp = len(pq.compress(image_to_byte_array(y_original)))
    print("this one is good 3")
    print("Size of Concatenated Image: {}".format(lenx_y))
    print("Size of x Image: {}".format(len_x_comp))
    print("Size of y Image: {}".format(len_y_comp))
    ncd = (lenx_y - min(len_x_comp, len_y_comp)) / \
    max(len_x_comp, len_y_comp)
    print("NCD of the same image using paq9 compression is: {}".format(ncd))



    print("________Compressed Images Tests_____________")

    # images are all mixed together even though they were only resized. what to do?
    print(com_images.shape) # all objects in the array are images
    a = com_images[0,:,:,:]
    print(com_images.shape)
    #list_images = com_images.tolist()
    #print(list_images)

    #heatmap kernel matrix
    # plot dist. size of images


    # was saving incorrectly the numpy array
    # https://medium.com/@muskulpesent/create-numpy-array-of-images-fecb4e514c4b

    # convert (number of images x height x width x number of channels) to (number of images x (height * width *3)) 
    # for example (120 * 40 * 40 * 3)-> (120 * 4800)
    
    #train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]*train.shape[3]])
