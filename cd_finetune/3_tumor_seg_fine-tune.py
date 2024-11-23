#!/usr/bin/env python
# coding: utf-8

from fastai.basics import *
from fastai.vision import models
from fastai.vision.all import *
from fastai.metrics import *
from fastai.data.all import *
from fastai.callback import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from semtorch import get_segmentation_learner
import torch


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        print(f"Directory '{dir_path}' already exists.")


#User input
#Dirs
proj_dir = '/fh/scratch/delete90/etzioni_r/lucas_l/michael_project/mutation_pred/'
train_file_location = proj_dir + '/intermediate_data/cd_finetune/cancer_detection_training/' #path to the train file
prior_model_location = proj_dir + '/models/cancer_detection_models/mets/' #path to the prior model
outdir = proj_dir + '/intermediate_data/cd_finetune/cancer_detection_training/' #path to where you want to save the new model


#Hyper-para
batch_size = 8
numeph = 100 #we don't need to train for too long 
lr = 0.000005 #super low learning rate
# weights = [1,1.5]  #this would be if we wanted to use weighted loss, currently not doing so
# w = torch.cuda.FloatTensor(weights)


#Check Cuda
if torch.cuda.is_available():
    device_index = torch.cuda.current_device()  # Get the current device index
    device_name = torch.cuda.get_device_name(device_index)
    print(f"CUDA Device: {device_name}")
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("CUDA is not available.")



#Load train file
#this is the training csv, please check the example
fns = pd.read_csv(train_file_location + 'training_file.csv')
codes = ['Background','Tumor']



#this datablock also sets augmentation, we can increase or decrease these as needed
segdata = DataBlock(blocks=(ImageBlock,MaskBlock),splitter=ColSplitter(),get_x=ColReader('img'),get_y=ColReader('label'),item_tfms=[Resize((250,250))],batch_tfms=[Normalize.from_stats(*imagenet_stats), Contrast(max_lighting = 0.2, p=0.9), Hue(max_hue = 0.1, p=0.9),Saturation(max_lighting=0.2, p=0.9)])
#segdata.summary(fns)
#this assumes device = 0 (usually the case for you)
dls=segdata.dataloaders(fns,bs=batch_size,tfm_y=True,device=device)
dls.show_batch(figsize=(12,12))


# this is the model name, you need to change every time or pass as an input 
model_name = 'dlv3_2ep_2e4_update-07182023_RT'
print(model_name)


# i like showing the batch so you can see how things look
plt.savefig(outdir + 'training_log/'+model_name+'_showbatch.png')


#nit learner
learn = get_segmentation_learner(dls=dls, number_classes=2, segmentation_type="Semantic Segmentation",
                                 cbs=[SaveModelCallback(fname=model_name),ShowGraphCallback()],
                                 architecture_name="deeplabv3+", backbone_name="resnet50",
                                 metrics=[Dice(), JaccardCoeff()],wd=1e-2)
# Here is where you load the prior model weights
learn.load(prior_model_location + '/dlv3_2ep_2e4_update-07182023_RT')
learn.to_fp16()


# we can also change up how many layers we freeze/fine-tune. can discuss in the future
learn.fit_one_cycle(numeph, lr)
#learn.to_fp32()


# save finished model
create_dir_if_not_exists(outdir + 'ft_models/')
learn.save(os.path.join(outdir,'ft_models',model_name + "_fine_tuned"))
learn.export(os.path.join(outdir,'ft_models',model_name +'_fine_tuned.pkl'))
learn.remove_cbs([SaveModelCallback])
learn.show_results(max_n=10, figsize=(7,8))
plt.savefig(outdir + 'training_log/'+ model_name + '_showresults_dice_deeplab.png')

