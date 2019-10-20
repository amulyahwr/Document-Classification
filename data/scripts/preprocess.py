import os
import shutil
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from PIL import Image
import torch
from sklearn import preprocessing

sss_train = StratifiedShuffleSplit(n_splits=1, test_size=0.003125, random_state=0)
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.0025, random_state=0)
sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.0025, random_state=0)
#copy first 1000 train data

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
    holistic = np.array(holistic)
    holistic = preprocessing.normalize(holistic)
    holistic = torch.from_numpy(holistic)
    # holistic = torch.unsqueeze(holistic, dim=0)
    # holistic = holistic.repeat(3, 1, 1)

    header = np.array(header)
    header = preprocessing.normalize(header)
    header = torch.from_numpy(header)
    # header = torch.unsqueeze(header, dim=0)
    # header = header.repeat(3, 1, 1)

    footer = np.array(footer)
    footer = preprocessing.normalize(footer)
    footer = torch.from_numpy(footer)
    # footer = torch.unsqueeze(footer, dim=0)
    # footer = footer.repeat(3, 1, 1)

    left_body = np.array(left_body)
    left_body = preprocessing.normalize(left_body)
    left_body = torch.from_numpy(left_body)
    # left_body = torch.unsqueeze(left_body, dim=0)
    # left_body = left_body.repeat(3, 1, 1)

    right_body = np.array(right_body)
    right_body = preprocessing.normalize(right_body)
    right_body = torch.from_numpy(right_body)
    # right_body = torch.unsqueeze(right_body, dim=0)
    # right_body = right_body.repeat(3, 1, 1)

    return holistic, header, footer, left_body, right_body


def copydata(path, sss, dst, labels):
    with open(path,'rU') as label_file:
        X = []
        y = []
        for label in label_file.readlines():
            X.append(label.split(' ')[0])
            y.append(int(label.split(' ')[1].rstrip()))

        X = np.array(X)
        y = np.array(y)

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
                              
#         for train_index, test_index in sss.split(X, y):
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]

        count = 0
        for image in tqdm(X, desc="Writing Tensors.."):
            old_path = os.path.join('../../stratified/rvl-cdip/images/',image)
            try:
                holistic, header, footer, left_body, right_body = preprocess_image(old_path)

                torch.save(holistic, dst+'holistic/%d.pt'%(count))
                torch.save(header, dst+'header/%d.pt'%(count))
                torch.save(footer, dst+'footer/%d.pt'%(count))
                torch.save(left_body, dst+'left_body/%d.pt'%(count))
                torch.save(right_body, dst+'right_body/%d.pt'%(count))
            except Exception as ex:
                print('Error:',ex)
                pass
            count = count + 1

        y = torch.from_numpy(y)
        torch.save(y, './labels/'+labels)
        # y_test = map(str, y_test)
        # y_test = '\n'.join(y_test)
        # with open('./labels/'+labels,'w') as write_labels:
        #     write_labels.write(y_test)

        print('Dataset size: %d'%(len(X_test)))

copydata('../../stratified/rvl-cdip/labels/train.csv', sss_train, './train/','train.pt')
copydata('../../stratified/rvl-cdip/labels/val.csv', sss_val, './val/','val.pt')
copydata('../../stratified/rvl-cdip/labels/test.csv', sss_test, './test/','test.pt')
