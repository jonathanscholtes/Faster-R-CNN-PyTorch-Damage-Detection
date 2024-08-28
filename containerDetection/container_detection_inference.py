import cv2
import numpy as np
import torch
from torchvision.ops import nms
from util.iou_extract import extract_candidates  
from model.dataset import preprocess_image

from model.FRCNN import FRCNN


class ContainerDetectionInference:
    def __init__(self, model_path, device=None):
        """
        Initialize the ContainerDetectionInference class.

        Args:
            model_path (str): Path to the trained model.
            device (str): Device to run the model on ('cpu' or 'cuda'). If None, auto-detect.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FRCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set model to evaluation mode


    def _process_image(self, file_name):
        """
        Process a single image to detect objects.

        Args:
            file_name (str): Path to the image file.

        Returns:
            dict: A dictionary containing bounding boxes and rotation angle.
        """
        img = cv2.imread(file_name, 1)[..., ::-1]  # Read and convert color space
        img = cv2.resize(img, (244, 244))  # Resize image
        H, W, _ = img.shape

        candidates = extract_candidates(img)
        candidates = [(x, y, x + w, y + h) for x, y, w, h in candidates]

        input_image = preprocess_image(img / 255.)[None]  # Preprocess image
        rois = np.array([[x, y, X, Y] for x, y, X, Y in candidates])
        rois = rois / np.array([W, H, W, H])
        rixs = np.array([0] * len(rois))

        rois, rixs = [torch.Tensor(item).to(self.device) for item in [rois, rixs]]

        with torch.inference_mode():
            probs, thetas, deltas = self.model(input_image, rois, rixs)
            confs, clss = torch.max(probs, -1)

        confs, clss, probs, thetas, deltas = [tensor.detach().cpu().numpy() for tensor in [confs, clss, probs, thetas, deltas]]
        candidates = np.array(candidates)

        ixs = clss != 0
        confs, clss, probs, thetas, deltas, candidates = [tensor[ixs] for tensor in [confs, clss, probs, thetas, deltas, candidates]]
        bbs = candidates + deltas

        ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
        confs, clss, probs, thetas, deltas, candidates, bbs = [tensor[ixs] for tensor in [confs, clss, probs, thetas, deltas, candidates, bbs]]

        if len(ixs) == 1:
            confs, clss, probs, thetas, deltas, candidates, bbs = [tensor[None] for tensor in [confs, clss, probs, thetas, deltas, candidates, bbs]]

        if len(bbs) > 0:
            bbs = bbs[0] / np.array([W, H, W, H])
            return {'bbs': [float(x) for x in bbs], 'theta': float(round(thetas[0][0], 2))}
        else:
            return {'bbs': [], 'theta': 0}


    def inference_model(self, files):
        """
        Run inference on a list of files.

        Args:
            files (list of str): List of paths to image files.

        Returns:
            list of dict: List of dictionaries containing bounding boxes and rotation angles for each image.
        """
        results = []
        for file_name in files:
            result = self._process_image(file_name)
            results.append(result)
        return results
