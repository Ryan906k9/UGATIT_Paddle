from scipy import misc
from PIL import Image
import imageio
import os, cv2
import numpy as np
import random




def load_test_data(image_path, size=256):
    #img = misc.imread(image_path, mode='RGB')
    img = imageio.imread(image_path)
    #img = misc.imresize(img, [size, size])
    img = np.array(Image.fromarray(img).resize((size, size)))
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def denorm(x):
    return x * 0.5 + 0.5

# 这里转 numpy 需要修改，具体RGB的顺序要实验了看
def tensor2numpy(x):
    #return x.detach().cpu().numpy().transpose(1,2,0)
    return x.numpy().transpose(1,2,0)

def RGB2BGR(x):
    #print(x.shape)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


# 图片预处理
# 随机水平翻转
def random_flip(img, thresh=0.5):
    img = np.asarray(img)
    if random.random() > thresh:
        img = img[ :, ::-1, : ]
    img = Image.fromarray(img)
    return img

# 统一图像大小
def img_resize(img, img_size=256):
    img = cv2.resize(img, (img_size + 30, img_size + 30))
    return img

# 随机裁剪
def random_crop(img, img_size=256):
    # if(random.random() > 0.5):
    #    return img
    img = np.asarray(img)
    m = random.randint(0, 30)
    n = random.randint(0, 30)
    # print(m,n)
    img = img[ m:(m + img_size), n:(n + img_size), : ]
    # print(img.shape)
    img = Image.fromarray(img)
    return img

# 归一化
def img_norm(img):
    img = np.asarray(img)
    img = img / 255
    img = (img - 0.5) / 0.5
    return img

def preprocess(img, img_size=256, arg='train'):

    if arg == 'train':
        img = img_resize(img)
        img = random_flip(img)
        img = random_crop(img)
        img = img_norm(img)

    if arg == 'test':
        img = img_resize(img)
        img = img_norm(img)

    return img

# 定义 reader
def custom_reader(file_dir):
    def reader():


        # 首先确定文件数量，文件名字的列表

        # 获取文件名
        file_names = os.listdir(file_dir)
        # print(file_names)

        # 文件名拼接路径
        file_list = [ os.path.join(file_dir, img_file) for img_file in file_names ]
        # print(file_list)

        for each in file_list:
            # 每次一次打开一个图片，获得其像素值矩阵
            img = cv2.imread(each)
            # 图片预处理
            img = preprocess(img)
            # 把 HWC 转成 CHW
            img = img[ :, :, ::-1 ].transpose((2, 0, 1))
            yield img  # 返回出来

    return reader