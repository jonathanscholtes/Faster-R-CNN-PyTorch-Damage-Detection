from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
from contextlib import redirect_stdout
import torch
import cv2
import numpy as np
from torchvision import transforms, models, datasets


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    return img.to(device).float()
     

class CoCoDataSet(Dataset):
    
    def __init__(self, path, annotations=None):
        super().__init__()

        self.path = os.path.expanduser(path)
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
 
        im = cv2.imread('{}/{}'.format(self.path, image),1)[...,::-1]

        boxes, categories = self._get_target(id)
        

        return im, boxes[0:4], categories, image, id


    def _get_target(self, id):
        'Get annotations for sample'

        ann_ids = self.coco.getAnnIds(imgIds=id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, categories = [], []
        for ann in annotations:
            final_bbox = ann['bbox']

            assert len(ann['bbox']) == 4, "Bounding box for id %i does not contain four entries." % id
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            if 'categories' in self.coco.dataset:
                cat = self.categories_inv[cat]
            categories.append(cat)

        return boxes , categories

class DamageDataset(Dataset):
    w, h = 224, 224
    def __init__(self,path, fpaths, labels, boxes, image_ids,transforms=None):
        self.fpaths = fpaths
        self.labels = labels
        self.boxes = boxes
        self.image_ids = image_ids
        self.path = os.path.expanduser(path)
        
    def __len__(self): return len(self.FPATHS)
    def __getitem__(self, ix):
       
        image_id = self.image_ids[ix]
        fpath = str(self.fpaths[ix])
        img = Image.open('{}/{}'.format(self.path, fpath)).convert("RGB")

        img = np.array(img)/255.0

        boxes = torch.from_numpy(np.array(self.boxes[ix]))


        labels = self.labels[ix]
        target = {}
        target["boxes"] = torch.Tensor(boxes).float()
        target["labels"] = torch.Tensor( labels).long()
        img = preprocess_image(img)
        return img, target

    def __len__(self) -> int:
        return len(self.fpaths)
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))