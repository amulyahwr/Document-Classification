import os
from tqdm import tqdm
import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, dataset_path, lst_files):
        super(Dataset, self).__init__()

        self.lst_files = lst_files
        self.dataset_path = dataset_path

        self.label_path = dataset_path.split('/')
        self.label_path.insert(2,'labels')
        self.label_path = '/'.join(self.label_path) + '.pt'

        self.labels = torch.load(self.label_path)


    def __len__(self):
        return len(self.lst_files)

    def __getitem__(self, lst_index):

        holistic_lst = []
        header_lst = []
        footer_lst = []
        left_body_lst = []
        right_body_lst = []
        labels = []

        for idx in lst_index:
            holistic = torch.load(self.dataset_path+'/holistic/%d.pt'%(idx))
            header = torch.load(self.dataset_path+'/header/%d.pt'%(idx))
            footer = torch.load(self.dataset_path+'/footer/%d.pt'%(idx))
            left_body = torch.load(self.dataset_path+'/left_body/%d.pt'%(idx))
            right_body = torch.load(self.dataset_path+'/right_body/%d.pt'%(idx))

            holistic = torch.unsqueeze(holistic, dim=0)
            holistic_lst.append(holistic)

            header = torch.unsqueeze(header, dim=0)
            header_lst.append(header)

            footer = torch.unsqueeze(footer, dim=0)
            footer_lst.append(footer)

            left_body = torch.unsqueeze(left_body, dim=0)
            left_body_lst.append(left_body)

            right_body = torch.unsqueeze(right_body, dim=0)
            right_body_lst.append(right_body)


            labels.append(torch.unsqueeze(self.labels[idx],dim=0))
        holistic = torch.cat(holistic_lst, dim=0)
        header = torch.cat(header_lst, dim=0)
        footer = torch.cat(footer_lst, dim=0)
        left_body = torch.cat(left_body_lst, dim=0)
        right_body = torch.cat(right_body_lst, dim=0)

        labels = torch.cat(labels)

        return holistic, header, footer, left_body, right_body, labels
