import os
from tqdm import tqdm
import torch
import torch.utils.data as data
from PIL import Image
import pandas as pd
import numpy as np
class Dataset(data.Dataset):
    def __init__(self, labels_path, model_type):
        super(Dataset, self).__init__()

        self.labels_path = labels_path
        self.data_container = 'data/stratified/rvl-cdip/images/'
        self.model_type  = model_type
#         self.label_path = dataset_path.split('/')
#         self.label_path.insert(2,'labels')
#         self.label_path = '/'.join(self.label_path) + '.pt'

        self.df_dataset = pd.read_csv(self.labels_path, sep=' ', names=['filenames','labels'])
#         print(self.labels_path, self.dataset_path)

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, lst_index):
        y_arr = []
        holistic_lst = []
        header_lst = []
        footer_lst = []
        left_body_lst = []
        right_body_lst = []
        labels = []
        filename = None

        for idx in lst_index:
            filename = os.path.join(self.data_container, self.df_dataset['filenames'][idx])

            im = Image.open(filename) # 375x500 # wxh
        #     im = im.resize((600, 780)) #width = 600, height =780


            holistic = np.array(im) # 500x375 # hxw #extracted regions as per: https://arxiv.org/pdf/1502.07058.pdf
            header = holistic[:(256*500)//780,:]
            footer = holistic[(524*500)//780:,:]
            left_body = holistic[(190*500)//780:(590*500)//780,:(300*375)//600]
            right_body = holistic[(190*500)//780:(590*500)//780,(300*375)//600:]

            #resizing as per: https://arxiv.org/pdf/1801.09321v3.pdf
#             holistic = np.array(Image.fromarray(holistic).resize((224, 224)))
#             header = np.array(Image.fromarray(header).resize((224, 224)))
#             footer = np.array(Image.fromarray(footer).resize((224, 224)))
#             left_body = np.array(Image.fromarray(left_body).resize((224, 224)))
#             right_body = np.array(Image.fromarray(right_body).resize((224, 224)))

            y_arr.append(self.df_dataset['labels'][idx])

#             holistic = torch.unsqueeze(holistic, dim=0)
            holistic_lst.append(holistic)

#             header = torch.unsqueeze(header, dim=0)
            header_lst.append(header)

#             footer = torch.unsqueeze(footer, dim=0)
            footer_lst.append(footer)

#             left_body = torch.unsqueeze(left_body, dim=0)
            left_body_lst.append(left_body)

#             right_body = torch.unsqueeze(right_body, dim=0)
            right_body_lst.append(right_body)


#             labels.append(torch.unsqueeze(self.labels[idx],dim=0))
        try:

            if 'vgg' in self.model_type:
                holistic = torch.from_numpy((np.array(holistic_lst)-127.5)/127.5)
                header = torch.from_numpy((np.array(header_lst)-127.5)/127.5)
                footer = torch.from_numpy((np.array(footer_lst)-127.5)/127.5)
                left_body = torch.from_numpy((np.array(left_body_lst)-127.5)/127.5)
                right_body = torch.from_numpy((np.array(right_body_lst)-127.5)/127.5)
            else:
                holistic = torch.from_numpy(np.array(holistic_lst)/255.0)
                header = torch.from_numpy(np.array(header_lst)/255.0)
                footer = torch.from_numpy(np.array(footer_lst)/255.0)
                left_body = torch.from_numpy(np.array(left_body_lst)/255.0)
                right_body = torch.from_numpy(np.array(right_body_lst)/255.0)
        except Exception as ex:
            print('Error:',ex)

        labels = torch.from_numpy(np.array(y_arr))

        return holistic, header, footer, left_body, right_body, labels
