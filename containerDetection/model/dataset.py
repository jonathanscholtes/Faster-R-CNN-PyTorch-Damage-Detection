from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from contextlib import redirect_stdout
import torch
import cv2
import numpy as np
from torchvision import transforms, models, datasets
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"device{device}")
print(torch.cuda.is_available())

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.to(device).float()
def decode(_y):
    _, preds = _y.max(-1)
    return preds

class CoCoDataSet(Dataset):
    
    def __init__(self, path, annotations=None):
        super().__init__()

        self.path = os.path.expanduser(path)
        self.annotation = annotations
        with redirect_stdout(None):
            self.coco = COCO(annotations)
        self.ids = list(self.coco.imgs.keys())
        if 'categories' in self.coco.dataset:
            self.categories_inv = {k: i for i, k in enumerate(self.coco.getCatIds())}


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ' Get sample'

        # Load image
        id = self.ids[index]        
        image = self.coco.loadImgs(id)[0]['file_name']
        im = cv2.imread('{}\{}'.format(self.path, image),1)[...,::-1]

        boxes, categories = self._get_target(id)
        
        #target = torch.cat([boxes, categories], dim=1)
        #bbs = [int(i) for i in boxes[0:4]]
        return im, [boxes[0:4]], categories,boxes[4], image


    def _get_target(self, id):
        'Get annotations for sample'

        ann_ids = self.coco.getAnnIds(imgIds=id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, categories = [], []
        for ann in annotations:
            final_bbox = ann['bbox']
            if len(final_bbox) == 4:
                final_bbox.append(0.0)  # add theta of zero.
            assert len(ann['bbox']) == 5, "Bounding box for id %i does not contain five entries." % id
            boxes = ann['bbox']
            cat = ann['category_id']
            if 'categories' in self.coco.dataset:
                cat = self.categories_inv[cat]
            categories.append(cat)


        return boxes , categories
    
    def split_into_chunks(self, num_chunks):
        
        chunk_size = math.ceil(len(self.ids) / num_chunks)

        chunks = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk_dataset = CoCoDataSet(
                self.path,
                annotations=self.annotation
            )
            chunk_dataset.ids = self.ids[start:end]
            chunk_dataset.coco = self.coco
            chunk_dataset.categories_inv = self.categories_inv if hasattr(self, 'categories_inv') else None
            chunks.append(chunk_dataset)
        
        return chunks

class ContainerDataset(Dataset):
    def __init__(self,path, fpaths, rois, labels, deltas, gtbbs,thetas):
        self.fpaths = fpaths
        self.gtbbs = gtbbs
        self.rois = rois
        self.labels = labels
        self.deltas = deltas
        self.thetas = thetas
        self.path = os.path.expanduser(path)
    def __len__(self): return len(self.fpaths)
    def __getitem__(self, ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread('{}/{}'.format(self.path, fpath), 1)[...,::-1]
        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        thetas = self.thetas[ix]
        assert len(rois) == len(labels) == len(deltas), f'{len(rois)}, {len(labels)}, {len(deltas)}'
        return image, rois, labels, deltas, gtbbs, fpath,thetas

    def collate_fn(self, batch):
        input, rois, rixs, labels, deltas,thetas = [], [], [], [], [],[]
        
        for ix in range(len(batch)):
            image, image_rois, image_labels, image_deltas, image_gt_bbs, image_fpath,image_thetas = batch[ix]
            image = cv2.resize(image, (244,244))
            input.append(preprocess_image(image/255.)[None])
            rois.extend(image_rois)
            rixs.extend([ix]*len(image_rois))
            labels.extend( image_labels)
            deltas.extend(image_deltas)
            thetas.extend(image_thetas)

        input = torch.cat(input).to(device)
        rois = torch.Tensor(rois).float().to(device)
        rixs = torch.Tensor(rixs).float().to(device)
        labels = torch.Tensor(labels).long().to(device)
        deltas = torch.Tensor(deltas).float().to(device)
        thetas = torch.Tensor(thetas).float().to(device)
        return input, rois, rixs, labels, deltas,thetas
    
    def split_into_chunks(self, chunk_size):

        num_chunks = math.ceil(len(self.fpaths) / chunk_size)
        chunks = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk_dataset = ContainerDataset(
                self.path,
                self.fpaths[start:end],
                self.rois[start:end],
                self.labels[start:end],
                self.deltas[start:end],
                self.gtbbs[start:end],
                self.thetas[start:end]
            )
            chunks.append(chunk_dataset)
        
        return chunks




