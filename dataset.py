import os
from tqdm import tqdm
import torch
import torch.utils.data as data
from PIL import Image
import pandas as pd
import numpy as np
class Dataset(data.Dataset):
    def __init__(self, labels_path, dataset_path, lst_files):
        super(Dataset, self).__init__()
        
        self.lst_files = lst_files
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.data_container = 'data'
        
#         self.label_path = dataset_path.split('/')
#         self.label_path.insert(2,'labels')
#         self.label_path = '/'.join(self.label_path) + '.pt'

        self.df_labels = pd.read_csv(self.labels_path, sep=' ')
#         print(self.labels_path, self.dataset_path)

    def __len__(self):
        return len(self.lst_files)

    def __getitem__(self, lst_index):
        y_arr = []
        holistic_lst = []
        header_lst = []
        footer_lst = []
        left_body_lst = []
        right_body_lst = []
        labels = []
        filename = None
#         print('lst_index:',lst_index)
        filename = os.path.join(self.data_container,self.dataset_path)
#         print('filename:',filename)
        for idx in lst_index:
           
     
            holistic_filename = os.path.join(filename,'holistic/%d.jpg'%(idx))
#             print('holistic_filename:',holistic_filename)
            header_filename = os.path.join(filename,'header/%d.jpg'%(idx))
            footer_filename = os.path.join(filename,'footer/%d.jpg'%(idx))
            left_body_filename = os.path.join(filename,'left_body/%d.jpg'%(idx))
            right_body_filename = os.path.join(filename,'right_body/%d.jpg'%(idx))
            
            filename_matcher = '/'+os.path.join(self.dataset_path,'holistic/%d.jpg'%(idx))
            
#             print('filename_matcher:',filename_matcher)
#             print(self.df_labels[self.df_labels['filenames']==filename_matcher]['labels'].values[0])
            
            y_arr.append(self.df_labels[self.df_labels['filenames']==filename_matcher]['labels'].values[0])
            
            holistic = np.array(Image.open(holistic_filename))
            header = np.array(Image.open(header_filename))
            footer = np.array(Image.open(footer_filename))
            left_body = np.array(Image.open(left_body_filename))
            right_body = np.array(Image.open(right_body_filename))

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
            
        holistic = torch.from_numpy(np.array(holistic_lst))
        header = torch.from_numpy(np.array(header_lst))
        footer = torch.from_numpy(np.array(footer_lst))
        left_body = torch.from_numpy(np.array(left_body_lst))
        right_body = torch.from_numpy(np.array(right_body_lst))

        labels = torch.from_numpy(np.array(y_arr))

        return holistic, header, footer, left_body, right_body, labels
