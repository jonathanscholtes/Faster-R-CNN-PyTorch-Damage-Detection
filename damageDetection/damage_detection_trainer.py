import json
import torch
import random
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.ops import nms
from torch.utils.tensorboard import SummaryWriter
from model.dataset import CoCoDataSet, DamageDataset
from torch_snippets.torch_loader  import Report
import mlflow
import mlflow.pytorch

class DamageDetectionTrainer:
    def __init__(self, annotations_path, image_path, experiment_name="DamageDetection",n_epochs=25, model_path="trained_models/frcnn_damage.pt", tracking_uri="http://localhost:5000",num_classes=3):
       
        self.annotations_path = annotations_path
        self.image_path = image_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_epochs = n_epochs
        self.writer = SummaryWriter()
        self.model_path = model_path
        self.num_classes = num_classes
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._load_data()

    def _load_data(self):     

        self.FPATHS = []
        self.LABELS = []
        self.BOXES = []
        self.IMAGE_IDS = []

        ds = CoCoDataSet(self.image_path, annotations=self.annotations_path)

        for ix, input in enumerate(ds):
            img, boxes, labels, fpath, image_id = input

            self.FPATHS.append(fpath)
            self.LABELS.append(labels)
            self.BOXES.append(boxes)
            self.IMAGE_IDS.append(image_id)

        print('Records:', len(self.FPATHS))

        n_train = int(len(self.FPATHS) * 0.8)
        n_test = len(self.FPATHS) - n_train

        self.train_ds = DamageDataset(self.image_path, self.FPATHS[:n_train], self.LABELS[:n_train],
                        self.BOXES[:n_train], self.IMAGE_IDS[:n_train])
        
        self.test_ds = DamageDataset(self.image_path, self.FPATHS[n_test:], self.LABELS[n_test:],
                       self.BOXES[n_test:], self.IMAGE_IDS[n_test:])
        
        self.train_loader = DataLoader(
        self.train_ds,
        batch_size=4,
        collate_fn=self.train_ds.collate_fn,
        drop_last=True
        )

        self.test_loader = DataLoader(
        self.test_ds,
        batch_size=4,
        collate_fn=self.test_ds.collate_fn,
        drop_last=True
        )

    def _decode(self, _y):
        """Decodes the predictions from the model."""
        _, preds = _y.max(-1)
        return preds
    
    def _get_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def _train_batch(self, inputs, model, optimizer):
        """Trains the model on a single batch of data."""
        model.train()
        input, targets = inputs
        input = list(image.to(self.device) for image in input)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        losses = model(input, targets)
        loss = sum(loss for loss in losses.values())
        loss.backward()
        optimizer.step()
        return loss, losses

    @torch.no_grad()
    def _validate_batch(self, inputs, model):
        """Validates the model on a single batch of data."""
        model.train() 
        input, targets = inputs
        input = list(image.to(self.device) for image in input)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        losses = model(input, targets)
        loss = sum(loss for loss in losses.values())
        return loss, losses

    def train_and_validate(self):
        """Runs the training and validation loops for multiple epochs."""
        if len(self.train_loader) > 0 and len(self.test_loader) > 0:
            model = self._get_model().to(self.device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
            
            log = Report(self.n_epochs)
            best_accuracy = 0

            for epoch in range(self.n_epochs):
                _n_train = len(self.train_loader)
                for ix, inputs in enumerate(self.train_loader):
                    loss, losses = self._train_batch(inputs, model, optimizer)
                    loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
                        [losses[k] for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg']]
                    pos = (epoch + (ix + 1) / _n_train)
                    self.writer.add_scalar("Loss/train", loss.item(), pos)
                    #self.writer.add_scalar("Accuracy/train", accs.mean(), pos)
                    log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss.item(), 
                    trn_regr_loss=regr_loss.item(), trn_objectness_loss=loss_objectness.item(),
                    trn_rpn_box_reg_loss=loss_rpn_box_reg.item(), end='\r')

                _n_test = len(self.test_loader)
                val_losses = []

                for ix, inputs in enumerate(self.test_loader):
                    loss, losses = self._validate_batch(inputs, model)
                    pos = (epoch + (ix + 1) / _n_test)
                    loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg = \
                    [losses[k] for k in ['loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg']]
            
                    val_losses.append(loss.item())
                    self.writer.add_scalar("Loss/val", loss.item(), pos)
                    #self.writer.add_scalar("Accuracy/val", accs.mean(), pos)
                    log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss.item(), 
                    val_regr_loss=regr_loss.item(), val_objectness_loss=loss_objectness.item(),
                    val_rpn_box_reg_loss=loss_rpn_box_reg.item(), end='\r')

                    if loss.item() <= min(val_losses):
                        torch.save(model.state_dict(), self.model_path)
                        #best_accuracy = accs.mean()

                log.report_avgs(epoch + 1)

            self.writer.close()
            log.plot_epochs('trn_loss,val_loss'.split(','))

            with mlflow.start_run():
                #mlflow.log_metric("accuracy", best_accuracy)
                mlflow.pytorch.log_state_dict(model.state_dict(), 'model')
        else:
            print(f'Test loader: {len(self.test_loader)}, Train loader: {len(self.train_loader)}')



