import numpy as np
import sys
import json
from torch_snippets import *
from model.dataset import CoCoDataSet, preprocess_image
from util.iou_extract import extract_candidates, extract_iou
import progressbar
import threading
import time


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread =threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

store_lock = threading.Lock()

class ContainerDataProcessor:
    """
    ContainerDataProcessor extracts and processes image regions and prepares the necessary data for training FRCNN model to detect containers in images.
        
    """
   
    def __init__(self, image_path, annotations_path, results_path, N=500, thread_count=8):
        """        
        Parameters:
        dataset_path (str): Path to the directory containing the dataset images.
        annotations_path (str): Path to the JSON file containing annotations for the images.
        N (int): The number of images to process. Defaults to 500.
        """
        # Store paths to dataset and annotations
        self.annotations = annotations_path
        self.image_path = image_path
        self.results_path = results_path

        # Set the number of images to process
        self.N = N
        
        self.thread_count = thread_count

        # Target labels: background = 0, container = 1
        self.target2label = ['background', 'container']
        
        # Initialize the dataset using the provided paths
        self.ds = CoCoDataSet(self.image_path, annotations=self.annotations)
        
         # Initialize lists to store processed data: file paths, ground truth boxes, class labels, 
        # deltas (coordinate adjustments), regions of interest (ROIs), IoUs, and rotation angles (thetas)
        self.FPATHS, self.GTBBS, self.CLSS, self.DELTAS, self.ROIS, self.IOUS, self.THETAS = [], [], [], [], [], [], []
        

    def process_dataset(self):
        max_value = min(len(self.ds), self.N)
        self.bar = progressbar.ProgressBar(widgets=self._bar_widget(max_value), max_value=max_value)
        
        threads = []
        
        batches= self.ds.split_into_chunks(self.thread_count)

        print(len(batches))
        print(len(batches[0]))
       
    
        for batch in batches:
            t=self._process_batch(args=batch)
            threads.append(t)
            

        for t in threads:
            t.join()

        print("Finalize")
        self._finalize_results()


    @threaded
    def _process_batch(self,args):
        batch = args
        print(batch)
        
        #self.ds
        for ix, (im, bbs, labels, theta, fpath) in enumerate(batch):
            time.sleep(.001)

            if ix == self.N:
                break
            
            H, W, _ = im.shape
            candidates = self._get_candidates(im)
            ious, rois, clss, deltas, thetas = self._process_candidates(candidates, bbs, theta, H, W)
            
            #store_lock.acquire()
            self._store_results(fpath, ious, rois, clss, deltas, bbs, thetas)
        
            self.bar.next()

            #store_lock.release()
          
    

    def _bar_widget(self, max_value):
        return  [
        'Processing: ', progressbar.Percentage(),
        ' ', progressbar.Bar(marker='*', left='[', right=']'),
        ' ', progressbar.Counter(), f'/{max_value}',
        ' ', progressbar.Timer(),
        ' ', progressbar.ETA()]
         

    def _chunk_dataset(self,iterable, chunk_size):
   
        dataset_list = list(enumerate(iterable))
        
        # Split the dataset list into chunks
        for i in range(0, len(dataset_list), chunk_size):
            yield dataset_list[i:i + chunk_size]


    def _get_candidates(self, im):
        candidates = extract_candidates(im)
        return np.array([(x, y, x+w, y+h) for x, y, w, h in candidates])

    def _process_candidates(self, candidates, bbs, theta, H, W):
        ious, rois, clss, deltas, thetas = [], [], [], [], []
        ious_matrix = np.array([[extract_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T

        for jx, candidate in enumerate(candidates):
            cx, cy, cX, cY = candidate
            candidate_ious = ious_matrix[jx]
            best_iou_at = np.argmax(candidate_ious)
            best_iou = candidate_ious[best_iou_at]
            best_bb = bbs[best_iou_at]

            if best_iou > 0.3:
                clss.append(1)
            else:
                clss.append(0)

            thetas.append(theta)
            delta = np.array([best_bb[0] - cx, best_bb[1] - cy, best_bb[2] - cX, best_bb[3] - cY]) / np.array([W, H, W, H])
            deltas.append(list(delta.astype(float)))
            rois.append(list((candidate / np.array([W, H, W, H])).astype(float)))

        return ious_matrix, rois, clss, deltas, thetas

    def _store_results(self, fpath, ious, rois, clss, deltas, bbs, thetas):
        self.FPATHS.append(fpath)
        self.IOUS.append(ious)
        self.ROIS.append(rois)
        self.CLSS.append(clss)
        self.DELTAS.append(deltas)
        self.GTBBS.append(bbs)
        self.THETAS.append(thetas)

    def _finalize_results(self):
        data_json = {
            'FPATHS': self.FPATHS,
            'GTBBS': self.GTBBS,
            'CLSS': self.CLSS,
            'DELTAS': list(self.DELTAS),
            'ROIS': self.ROIS,
            'THETAS': self.THETAS
        }
        with open(self.results_path, 'w') as f:
            json.dump(data_json, f)


