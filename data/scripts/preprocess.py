import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from sklearn import preprocessing
import pandas as pd
import glob

orig_images_path = '../stratified/rvl-cdip/images/'
orig_train_labels_path = '../stratified/rvl-cdip/labels/train.csv'
orig_val_labels_path = '../stratified/rvl-cdip/labels/val.csv'
orig_test_labels_path = '../stratified/rvl-cdip/labels/test.csv'

def preprocess_image(image):

    im = Image.open(image) # 375x500 # wxh
#     im = im.resize((600, 780)) #width = 600, height =780


    holistic = np.array(im) # 500x375 # hxw #extracted regions as per: https://arxiv.org/pdf/1502.07058.pdf
    header = holistic[:(256*500)//780,:]
    footer = holistic[(524*500)//780:,:]
    left_body = holistic[(190*500)//780:(590*500)//780,:(300*375)//600]
    right_body = holistic[(190*500)//780:(590*500)//780,(300*375)//600:]

    #resizing as per: https://arxiv.org/pdf/1801.09321v3.pdf
    holistic = Image.fromarray(holistic).resize((224, 224))
    header = Image.fromarray(header).resize((224, 224))
    footer = Image.fromarray(footer).resize((224, 224))
    left_body = Image.fromarray(left_body).resize((224, 224))
    right_body = Image.fromarray(right_body).resize((224, 224))

    #normalizing, adding a dimension, replicating for 3 channels, converting to tensors,
    # holistic = np.array(holistic)
    # holistic = preprocessing.normalize(holistic)

    # holistic = torch.from_numpy(holistic)
    # holistic = torch.unsqueeze(holistic, dim=0)
    # holistic = holistic.repeat(3, 1, 1)

    # header = np.array(header)
    # header = preprocessing.normalize(header)
    # header = torch.from_numpy(header)
    # header = torch.unsqueeze(header, dim=0)
    # header = header.repeat(3, 1, 1)

    # footer = np.array(footer)
    # footer = preprocessing.normalize(footer)
    # footer = torch.from_numpy(footer)
    # footer = torch.unsqueeze(footer, dim=0)
    # footer = footer.repeat(3, 1, 1)

    # left_body = np.array(left_body)
    # left_body = preprocessing.normalize(left_body)
    # left_body = torch.from_numpy(left_body)
    # left_body = torch.unsqueeze(left_body, dim=0)
    # left_body = left_body.repeat(3, 1, 1)

    # right_body = np.array(right_body)
    # right_body = preprocessing.normalize(right_body)
    # right_body = torch.from_numpy(right_body)
    # right_body = torch.unsqueeze(right_body, dim=0)
    # right_body = right_body.repeat(3, 1, 1)

    return holistic, header, footer, left_body, right_body


def copydata(path, dst, labels):
    #with open(path,'rU') as label_file:
        data = pd.read_csv(path, names=['filenames','labels'],sep=' ')

        # X = []
        # y = []
        # for label in label_file.readlines():
        #     X.append(label.split(' ')[0])
        #     y.append(int(label.split(' ')[1].rstrip()))

        X = np.array(data['filenames'])
        y = np.array(data['labels'])

        if not os.path.exists(dst+'holistic/'):
            os.makedirs(dst+'holistic/')
        if not os.path.exists(dst+'header/'):
            os.makedirs(dst+'header/')
        if not os.path.exists(dst+'footer/'):
            os.makedirs(dst+'footer/')
        if not os.path.exists(dst+'left_body/'):
            os.makedirs(dst+'left_body/')
        if not os.path.exists(dst+'right_body/'):
            os.makedirs(dst+'right_body/')

        count = 0
        for image in tqdm(X, desc="Writing Tensors.."):
            old_path = os.path.join(orig_images_path,image)
            try:
                holistic, header, footer, left_body, right_body = preprocess_image(old_path)
                holistic.save(dst+'holistic/%d.jpg'%(count))
                header.save(dst+'header/%d.jpg'%(count))
                footer.save(dst+'footer/%d.jpg'%(count))
                left_body.save(dst+'left_body/%d.jpg'%(count))
                right_body.save(dst+'right_body/%d.jpg'%(count))

                # torch.save(holistic, dst+'holistic/%d.pt'%(count))
                # torch.save(header, dst+'header/%d.pt'%(count))
                # torch.save(footer, dst+'footer/%d.pt'%(count))
                # torch.save(left_body, dst+'left_body/%d.pt'%(count))
                # torch.save(right_body, dst+'right_body/%d.pt'%(count))
            except Exception as ex:
                print('Error:',ex)
                pass
            count = count + 1

        filenames = np.array(sorted(glob.glob(dst+'holistic/*.jpg')))

        df = pd.DataFrame(columns=['filenames','labels'])
        df['filenames'] = filenames
        df['labels'] = y

        #y = torch.from_numpy(y)

        if not os.path.exists('../labels'):
            os.makedirs('../labels')
        df.to_csv('../labels/'+labels, sep=' ', index=False)
        #torch.save(, '../labels/'+labels)

        print('Dataset size: %d'%(len(X)))

copydata(orig_train_labels_path, '../train/','train.csv')
copydata(orig_val_labels_path, '../val/','val.csv')
copydata(orig_test_labels_path, '../test/','test.csv')
