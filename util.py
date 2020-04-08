import os
import tarfile
import urllib.request

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import h5py

def is_png(path):
    return os.path.splitext(path)[1] == ".png"

def download_dataset(url):
    file_name,_=os.path.splitext(os.path.basename(url))
    dir_path = os.path.join(os.getcwd(), "storage")
    os.makedirs(dir_path, exist_ok=True)

    full_path=os.path.join(dir_path, file_name)

    if not os.path.exists(full_path):
        urllib.request.urlretrieve(url,filename=full_path)

    return full_path

def show_directories(path):
    with tarfile.open(path,"r") as tar:
        for name in tar.getnames():
            if not is_png(name):
                print(name)

def extract_dataset(path):
    dir_name="dataset_{}".format(os.path.basename(path).split(".")[0])
    dir_path=os.path.join(os.path.dirname(path), dir_name)

    if not os.path.exists(dir_path):
        with tarfile.open(path,"r") as tar:
            tar.extractall(dir_path)           
    
    return dir_path

def read_dataset(path,classes,img_width,img_height):
    n=img_height*img_width

    X=[]
    Y=[]

    for root, _, files in os.walk(path):
        for file in files:
            try:
                im=mpimg.imread(os.path.join(root, file))
                X.append(im.reshape(1,n).T)

                dir_name=os.path.basename(root)
                Y.append(classes.index(dir_name))
            except:
                pass

    m=len(X)
    X=np.array(X).T.reshape((n,m))
    Y=np.array(Y).T.reshape((1,m))

    return X,Y

def show_images(X,Y,classes,img_height,img_width):
    m=X.shape[1]
    rand_index=np.random.randint(0,m,25)

    plt.figure(figsize=(10,10))
    for i in range(len(rand_index)):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[:,rand_index[i]].reshape(img_height,img_width),cmap=plt.cm.binary)
        plt.xlabel(classes[Y[0,rand_index[i]]])
    plt.show()

def show_percentages(Y,classes):
    total=Y.shape[1]
    for i in range(len(classes)):
        count=np.count_nonzero(Y==i)
        print("{0} : {1:.2f}%".format(classes[i],count/total*100))

def split_dataset(X,Y,train_size, valid_size,test_size):    
    train_index=train_size
    valid_index=train_index+valid_size
    test_index=valid_index+test_size

    p=np.random.permutation(X.shape[1])

    X_split=np.hsplit(X[:,p], [train_index,valid_index,test_index])
    Y_split=np.hsplit(Y[:,p], [train_index,valid_index,test_index])
    return X_split[0],X_split[1],X_split[2],Y_split[0],Y_split[1],Y_split[2]

def get_files(path):
    file_list=[]
    for root, _, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    
    return file_list


def create_hdf5(path,classes,img_width,img_height):
    file_name="{}.hdf5".format(os.path.basename(path))
    file_path=os.path.join(os.path.dirname(path), file_name)

    if os.path.exists(file_path):
        return file_path

    addrs = get_files(path)

    X_shape=(len(addrs),img_width,img_height)
    Y_shape=(len(addrs),1)



    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset("X",X_shape,np.float)
        hf.create_dataset("Y",Y_shape,np.uint8)

        for i in range(X_shape[0]):
            try:
                im=mpimg.imread(addrs[i])
                hf["X"][i,...]=im

                dir_name=os.path.basename(os.path.dirname(addrs[i]))
                hf["Y"][i]=classes.index(dir_name)
            except:
                pass                   

    return file_path

def read_hdf5(path):
    with h5py.File(path, 'r') as hf:
        return np.array(hf["X"]), np.array(hf["Y"])
