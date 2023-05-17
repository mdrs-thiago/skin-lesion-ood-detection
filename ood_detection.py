import pandas as pd 
import numpy as np 
import os 
from glob import glob

from PIL import Image 

import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm 

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from utils import FocalLoss
from model_utils import HAM10000, CustomDataset, train
from ood_metrics import get_measures
from ood_methods import OpenPCS, Mahalanobis, msp_get_scores, energy_get_scores


def load_dataset():
    df = pd.read_csv('HAM10000_metadata')

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    df['image_id'] += '.jpg'

    ood_classes = ['df','vasc','akiec']
    ood_df = df[df['dx'].isin(ood_classes)].copy()
    ood_df['cell_type_idx'] = pd.Categorical(ood_df['cell_type']).codes
    ood_df.reset_index(inplace=True)
    df_in = df.drop(df[df['dx'].isin(ood_classes)].index).copy()
    df_in['cell_type_idx'] = pd.Categorical(df_in['cell_type']).codes
    df_train, df_val = train_test_split(df_in, test_size=0.2)
    df_train.reset_index(inplace=True)
    df_val.reset_index(inplace=True)
    
    return df_train, df_val, df_in, ood_df

def train_model(df_train, df_val, device='cuda', model_name = 'google/vit-base-patch16-224'):
    
    extractor = AutoFeatureExtractor.from_pretrained(model_name);

    model = AutoModelForImageClassification.from_pretrained(model_name);

    for param in model.parameters():
        param.requires_grad = False


    if model_name == 'microsoft/resnet-50':
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, classes)

    mean = extractor.image_mean
    std = extractor.image_std
    if model_name == "microsoft/resnet-50" or model_name == "facebook/convnext-tiny-224":
        input_size = (extractor.size['shortest_edge'], extractor.size['shortest_edge'])
    else:
        input_size = (extractor.size['height'], extractor.size['width'])

    train_transform = transforms.Compose([transforms.Resize(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(20),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    val_transform = transforms.Compose([transforms.Resize(input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])    

    train_ds = HAM10000(df_train, transform=train_transform)
    val_ds = HAM10000(df_val, transform=val_transform)

    batch_size = 32
    epochs = 20
    lr = 1e-3

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,shuffle=False)

    model.to(device);

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    file_path = os.path.join('checkpoints',f"{model_name.replace('/','-')}.pth") 
    if os.path.isfile(file_path):
        model = model.load_state_dict(torch.load(file_path))
    else:
        train(model, train_loader, val_loader, optimizer, criterion, epochs)
        torch.save(model.state_dict(), file_path)

    return model, train_transform, val_transform

def print_results(id_features, ood_features, ood_features2, ood_features3, ood_features4):
    print(f'Experiment 1: ')
    auroc, aupr_in, aupr_out, fpr, fpr_, tpr_ = get_measures(id_features.reshape(-1,1), ood_features.reshape(-1,1))
    print(f'AUROC = {auroc}, AUPR = {aupr_in}, FPR95 = {fpr}')
    print(f'{round(auroc,4)} & {round(aupr_in,4)} & {round(fpr,4)}')

    print(f'Experiment 2: ')
    auroc, aupr_in, aupr_out, fpr, fpr_, tpr_ = get_measures(id_features.reshape(-1,1), ood_features2.reshape(-1,1))
    print(f'AUROC = {auroc}, AUPR = {aupr_in}, FPR95 = {fpr}')
    print(f'{round(auroc,4)} & {round(aupr_in,4)} & {round(fpr,4)}')


    print(f'Experiment 3: ')
    auroc, aupr_in, aupr_out, fpr, fpr_, tpr_ = get_measures(id_features.reshape(-1,1), ood_features3.reshape(-1,1))
    print(f'AUROC = {auroc}, AUPR = {aupr_in}, FPR95 = {fpr}')
    print(f'{round(auroc,4)} & {round(aupr_in,4)} & {round(fpr,4)}')


    print(f'Experiment 4: ')
    auroc, aupr_in, aupr_out, fpr, fpr_, tpr_ = get_measures(id_features.reshape(-1,1), ood_features4.reshape(-1,1))
    print(f'AUROC = {auroc}, AUPR = {aupr_in}, FPR95 = {fpr}')
    print(f'{round(auroc,4)} & {round(aupr_in,4)} & {round(fpr,4)}')


if __name__ == '__main__':
    df_train, df_val, df_in, ood_df = load_dataset()

    classes = df_in['cell_type_idx'].nunique()
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model_name = 'google/vit-base-patch16-224'
    

    model, train_transform, val_transform = train_model(df_train, df_val, device=device, model_name=model_name)

    batch_size = 1

    train_ds = HAM10000(df_train, transform=train_transform)
    in_ds = HAM10000(df_val, transform=val_transform)
    out_ds = HAM10000(ood_df, transform=val_transform)
    out_ds2 = CustomDataset('ood_lesions', transform=val_transform)
    out_ds3 = CustomDataset('ood_monkeypox', transform=val_transform)
    out_ds4 = CustomDataset('ood_other_lesions', transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    id_loader = DataLoader(in_ds, batch_size=batch_size, shuffle=True)
    ood_loader = DataLoader(out_ds, batch_size=batch_size,shuffle=False)
    ood_loader2 = DataLoader(out_ds2, batch_size=batch_size,shuffle=False)
    ood_loader3 = DataLoader(out_ds3, batch_size=batch_size,shuffle=False)
    ood_loader4 = DataLoader(out_ds4, batch_size=batch_size,shuffle=False)



    openpcs = OpenPCS(model_name, model, device, n_components=400)
    np_features, np_labels = openpcs.get_id_features(train_loader)
    id_features, _ = openpcs.get_id_features(id_loader)


    ood_features = openpcs.get_ood_features(model, ood_loader)
    ood_features2 = openpcs.get_ood_features(model, ood_loader2)
    ood_features3 = openpcs.get_ood_features(model, ood_loader3)
    ood_features4 = openpcs.get_ood_features(model, ood_loader4)

    openpcs.fit_PCA(in_scores = np_features, labels=np_labels)

    in_scores = openpcs.get_scores(id_features)
    ood_scores = openpcs.get_scores(ood_features)
    ood_scores2 = openpcs.get_scores(ood_features2)
    ood_scores3 = openpcs.get_scores(ood_features3)
    ood_scores4 = openpcs.get_scores(ood_features4)

    print_results(in_scores, ood_scores,ood_scores2, ood_scores3, ood_scores4)
   
    maha = Mahalanobis(n_components=1)

    maha.fit_PCA(in_scores = np_features, labels=np_labels)

    in_scores = maha.get_scores(id_features)
    ood_scores = maha.get_scores(ood_features)
    ood_scores2 = maha.get_scores(ood_features2)
    ood_scores3 = maha.get_scores(ood_features3)
    ood_scores4 = maha.get_scores(ood_features4)

    print_results(in_scores, ood_scores,ood_scores2, ood_scores3, ood_scores4)

    msp_id_features = msp_get_scores(model, id_loader)
    msp_ood_features = msp_get_scores(model, ood_loader)
    msp_ood_features2 = msp_get_scores(model, ood_loader2)
    msp_ood_features3 = msp_get_scores(model, ood_loader3)
    msp_ood_features4 = msp_get_scores(model, ood_loader4)

    print_results(msp_id_features, msp_ood_features, msp_ood_features2, msp_ood_features3, msp_ood_features4)

    en_id_features = energy_get_scores(model, id_loader)
    en_ood_features = energy_get_scores(model, ood_loader)
    en_ood_features2 = energy_get_scores(model, ood_loader2)
    en_ood_features3 = energy_get_scores(model, ood_loader3)
    en_ood_features4 = energy_get_scores(model, ood_loader4)

    print_results(en_id_features, en_ood_features, en_ood_features2, en_ood_features3, en_ood_features4)
