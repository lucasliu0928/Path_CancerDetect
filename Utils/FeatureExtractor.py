import torch
import timm
import os
import torch.nn as nn
import ResNet as ResNet
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from Utils import generate_deepzoom_tiles, extract_tile_start_end_coords_tma
    
class PretrainedModelLoader:
    def __init__(self, model_name, model_path, device='cpu'):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        if self.model_name == 'retccl':
            return self._load_retccl()
        elif self.model_name == 'uni1':
            return self._load_uni1()
        elif self.model_name == 'uni2':
            return self._load_uni2()
        elif self.model_name == 'prov_gigapath':
            return self._load_prov_gigapath()
        else:
            raise ValueError(f"Unknown feature extraction method: {self.model_name}")

    def _load_retccl(self):
        model = ResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
        pretext_model = torch.load(os.path.join(self.model_path,'best_ckpt.pth'),map_location=self.device)
        model.fc = nn.Identity()
        model.load_state_dict(pretext_model, strict=True)
        return model

    def _load_uni1(self):
        model = timm.create_model("vit_large_patch16_224",img_size = 224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True) # img_size=224, patch_size=16, 
        model.load_state_dict(torch.load(os.path.join(self.model_path, "vit_large_patch16_224.dinov2.uni_mass100k.bin"), map_location=self.device), strict=True)
        return model

    def _load_uni2(self):
        timm_kwargs = {
           'model_name': 'vit_giant_patch14_224',
           'img_size': 224, 
           'patch_size': 14, 
           'depth': 24,
           'num_heads': 24,
           'init_values': 1e-5, 
           'embed_dim': 1536,
           'mlp_ratio': 2.66667*2,
           'num_classes': 0, 
           'no_embed_class': True,
           'mlp_layer': timm.layers.SwiGLUPacked, 
           'act_layer': torch.nn.SiLU, 
           'reg_tokens': 8, 
           'dynamic_img_size': True
          }
        model = timm.create_model(**timm_kwargs)
        model.load_state_dict(torch.load(os.path.join(self.model_path, "uni2-h.bin"), map_location=self.device), strict=True)
        return model

    def _load_prov_gigapath(self):
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        return model




class TileEmbeddingExtractor(Dataset):
    def __init__(self, tile_info, image, pretrain_model_name, pretrain_model, device, image_type = 'WSI'):
        r'''
        Given a dataframe contains tiles info
        Return grabbed tiles, and embeddings
        '''
        super().__init__()
        self.tile_info = tile_info
        self.image = image
        self.mag_extract = list(set(tile_info['MAG_EXTRACT']))[0]
        self.save_image_size = list(set(tile_info['SAVE_IMAGE_SIZE']))[0]
        self.pixel_overlap = list(set(tile_info['PIXEL_OVERLAP']))[0]
        self.device = device
        self.pretrain_model = pretrain_model.to(self.device)
        self.transform = self._transform_functions(pretrain_model_name)
        self.image_type = image_type


    def __getitem__(self, idx):
        #Get x, y index
        tile_ind = self.tile_info['TILE_XY_INDEXES'].iloc[idx].strip("()").split(", ")
        x ,y = int(tile_ind[0]) , int(tile_ind[1])

        #Pull tiles
        if self.image_type == 'WSI':
            tile_pull = self._pull_tile(x ,y)
        elif self.image_type == 'TMA':
            tile_pull = self._pull_tile_tma(x ,y)
            
        #Get features
        features = self._get_embedding(tile_pull)
        
        return tile_pull,features

    def _pull_tile(self, x, y):
        #Generate tiles
        tiles, tile_lvls, _ , _ = generate_deepzoom_tiles(self.image,self.save_image_size, self.pixel_overlap, limit_bounds=True)  

        #pull tile
        tile_pull = tiles.get_tile(tile_lvls.index(self.mag_extract), (x, y))

        #resize 
        tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size),resample=PIL.Image.LANCZOS) 
        
        return tile_pull

    def _pull_tile_tma(self, x, y):
        #Pull tiles
        tile_starts, tile_ends, save_coords, tile_coords = extract_tile_start_end_coords_tma(x, y, tile_size = self.save_image_size, overlap = self.pixel_overlap)
        tile_pull = self.image.crop(box=(tile_starts[0], tile_starts[1], tile_ends[0], tile_ends[1])) 
        tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size),resample=PIL.Image.LANCZOS) #resize
        tile_pull = tile_pull.convert('RGB')

        return tile_pull
        

    def _get_embedding(self, tile_pull):
        #Transform tile image
        tile_pull_trns = self.transform(tile_pull)
        tile_pull_trns = tile_pull_trns.unsqueeze(0).to(self.device)  # Adds a dimension 
        
        #Use model to get feature
        self.pretrain_model.eval()
        with torch.no_grad():
            features = self.pretrain_model(tile_pull_trns)
        
        return features.cpu().numpy()
        

    def _transform_functions(self, model_name):
    
        if model_name == 'retccl':
            trnsfrms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.485, 0.456, 0.406), std =(0.229, 0.224, 0.225))
                ]
            )
        elif model_name == 'uni1' or model_name == 'uni2':
            trnsfrms = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        elif model_name == 'prov_gigapath':
            trnsfrms = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
    
        return trnsfrms