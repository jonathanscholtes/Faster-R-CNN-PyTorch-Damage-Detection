from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch
import os
import numpy as np
import cv2
from torchvision.ops import nms

class DamageDetectionInference:
    def __init__(self, model_path, device=None,num_classes=3,target2label = ['background','damage','tag']):

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.target2label = target2label
        self.model = self._get_model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()


    def _get_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model.to(self.device)
    
    def _permute_image(self,img):
        img = torch.tensor(img).permute(2,0,1)
        return img.to(self.device).float()
    
    def _decode_output(self,output):
        
        'convert tensors to numpy arrays'
        bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    
        labels = np.array([self.target2label[i] for i in output['labels'].cpu().detach().numpy()])
        confs = output['scores'].cpu().detach().numpy()
        ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
        bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

        if len(ixs) == 1:
            bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
        return bbs.tolist(), confs.tolist(), labels.tolist()

    def _process_image(self, file_name):
        img_orig =cv2.imread(file_name)
        outputs = self.model([self._permute_image(img_orig/255.0)])
        tags = []
        damage = []

        for ix, output in enumerate(outputs):
            bbs, confs, labels = self._decode_output(output)
            
            if(len(bbs)>0):
                for i in range(len(bbs)):
                    bbox = bbs[i]
                    lbl = labels[i]
                    x1,y1,x2,y2 = bbox
                    
                    if lbl == 'tag':
                        tags.append({'bbs':[x1,y1,x2,y2 ]})
                    if lbl == 'damage':
                        damage.append({'bbs':[x1,y1,x2,y2 ]})
        return {'damage':damage,'tags':tags}

    def _draw_bbox(self,img,bbox,draw_color):
        x1,y1,x2,y2 = bbox
        w = (x2-x1)
        x = (x1+x2)/2
        y=(y1+y2)/2
        h = (y2-y1) 
        rot_rectangle = ((x, y), (w, h), 0)        
        box = cv2.boxPoints(rot_rectangle) 
        box = np.intp(box)
        img = cv2.drawContours(img,[box],0,draw_color,2)
        return img
        
         
    def _write_text(self,img,text,bbox,draw_color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = .5
        font_thickness = 1
        x1,y1,x2,y2 = bbox

        img =cv2.putText(img, text, (int(x1),int(y1)), font, font_size, draw_color, font_thickness)
        return img

    def _draw_bounding_boxes_and_save(self,image_path,results,output_path):
        
        damage_color = (0,0,255)
        tag_color = (0,100,100)
        

        print(image_path)
        img = cv2.imread(image_path)

        for r in results['damage']:
            bbs = r['bbs']
            img = self._draw_bbox(img,bbs,damage_color)
            img = self._write_text(img,'damage',bbs,damage_color)


        for r in results['tags']:
            bbs = r['bbs']
            img = self._draw_bbox(img,bbs,tag_color)
            img = self._write_text(img,'tag',bbs,tag_color)

        cv2.imwrite(output_path, img)
           
    def inference_model(self, files, output_path):
        results = []        

        for file_path in files:
            file_name = os.path.basename(file_path)
            result = self._process_image(file_path)
            result['image_path']=str(file_path)

            if output_path != None and os.path.isdir(output_path):
                output_file = os.path.join(output_path, file_name)
                self._draw_bounding_boxes_and_save(file_path,
                                                 result,                 
                                                output_file)
                result['output_file']=output_file
                
            
            results.append(result)
        return results
