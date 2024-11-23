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

#check cuda
print(torch.cuda.is_available())

numeph = 10 #we don't need to train for too long 
lr = 0.000005 #super low learning rate
# weights = [1,1.5]  #this would be if we wanted to use weighted loss, currently not doing so
# w = torch.cuda.FloatTensor(weights)

#path to where you want to save the new model
outdir = '/data/MIP/harmonsa/pca_fhcrc/tumor_seg/training_log'

#this is the training csv, please check the example
fns = pd.read_csv('/data/MIP/harmonsa/pca_fhcrc/tumor_seg/train_july2.csv')
codes = ['Background','Tumor']

#this datablock also sets augmentation, we can increase or decrease these as needed
segdata = DataBlock(blocks=(ImageBlock,MaskBlock),splitter=ColSplitter(),get_x=ColReader('img'),get_y=ColReader('label'),item_tfms=[Resize((250,250))],batch_tfms=[Normalize.from_stats(*imagenet_stats), Contrast(max_lighting = 0.2, p=0.9), Hue(max_hue = 0.1, p=0.9),Saturation(max_lighting=0.2, p=0.9)])
segdata.summary(fns)
#this assumes device = 0 (usually the case for you)
dls=segdata.dataloaders(fns,bs=128,tfm_y=True,device=torch.device("cuda:0"))
dls.show_batch(figsize=(12,12))

# this is the model name, you need to change every time or pass as an input 
model_name = 'dlv3_RT'
print(model_name)
# i like showing the batch so you can see how things look
plt.savefig('/data/MIP/harmonsa/pca_fhcrc/tumor_seg/training_log/'+model_name+'_showbatch.png')


learn = get_segmentation_learner(dls=dls, number_classes=2, segmentation_type="Semantic Segmentation",
                                 cbs=[SaveModelCallback(fname=model_name)],
                                 architecture_name="deeplabv3+", backbone_name="resnet50",
                                 metrics=[Dice(), JaccardCoeff()],wd=1e-2)


# Here is where you load the prior model weights
learn.load('/data/MIP/harmonsa/pca_fhcrc/tumor_seg/training_log/saved_models/dlv3_5ep_1e4_update')
learn.to_fp16()
# we can also change up how many layers we freeze/fine-tune. can discuss in the future
learn.fit_one_cycle(numeph, lr)
learn.to_fp32()
# save finished model
learn.save(os.path.join(outdir,'saved_models',model_name))
learn.export(os.path.join(outdir,'exported_models',model_name+'.pkl'))
learn.remove_cbs([SaveModelCallback])
learn.show_results(max_n=6, figsize=(7,8))
plt.savefig('/data/MIP/harmonsa/pca_fhcrc/tumor_seg/training_log/'+model_name+'_showresults_dice_deeplab.png')

