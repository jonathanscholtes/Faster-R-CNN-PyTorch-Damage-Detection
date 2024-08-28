import json
import torch
import random
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.dataset import ContainerDataset
from model.FRCNN import FRCNN
from torch_snippets.torch_loader  import Report
import mlflow
import mlflow.pytorch

class ContainerDetectionTrainer:
    def __init__(self, annotations_path, image_path, data_path, experiment_name="ContainerDetection",n_epochs=25, model_path="trained_models/frcnn_container.pt", tracking_uri="http://localhost:5000"):
        """
        Initializes the trainer with dataset paths and training settings.

        Parameters:
        annotations_path (str): Path to the annotations file (COCO format).
        image_path (str): Path to the directory containing images.
        experiment_name (str): MLflow experiment name.
        tracking_uri (str): MLflow tracking URI.
        """
        self.annotations_path = annotations_path
        self.image_path = image_path
        self.data_path = data_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_epochs = n_epochs
        self.writer = SummaryWriter()
        self.model_path = model_path
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._load_data()

    def _load_data(self):
        """Loads the training data from the annotation file."""
        with open(self.data_path, 'r') as f:
            data_json = json.load(f)

        self.FPATHS = data_json['FPATHS']
        self.GTBBS = data_json['GTBBS']
        self.CLSS = data_json['CLSS']
        self.DELTAS = data_json['DELTAS']
        self.ROIS = data_json['ROIS']
        self.THETAS = data_json['THETAS']

        print('Records:', len(self.FPATHS))

        n_train = int(len(self.FPATHS) * 0.8)
        n_test = len(self.FPATHS) - n_train

        self.train_ds = ContainerDataset(self.image_path, self.FPATHS[:n_train], self.ROIS[:n_train],
                                         self.CLSS[:n_train], self.DELTAS[:n_train], self.GTBBS[:n_train], self.THETAS[:n_train])

        self.test_ds = ContainerDataset(self.image_path, self.FPATHS[n_test:], self.ROIS[n_test:],
                                        self.CLSS[n_test:], self.DELTAS[n_test:], self.GTBBS[n_test:], self.THETAS[n_test:])

        self.train_loader = DataLoader(self.train_ds, batch_size=6, collate_fn=self.train_ds.collate_fn, drop_last=True)
        self.test_loader = DataLoader(self.test_ds, batch_size=6, collate_fn=self.test_ds.collate_fn, drop_last=True)

    def _decode(self, _y):
        """Decodes the predictions from the model."""
        _, preds = _y.max(-1)
        return preds

    def _train_batch(self, inputs, model, optimizer, criterion):
        """Trains the model on a single batch of data."""
        input, rois, rixs, clss, deltas, thetas = inputs
        model.train()
        optimizer.zero_grad()
        _clss, _theta_score, _deltas = model(input, rois, rixs)
        loss, loc_loss, regr_loss, theta_loss = criterion(_clss, _theta_score, _deltas, clss, thetas.view(-1, 1), deltas)
        accs = clss == self._decode(_clss)
        loss.backward()
        optimizer.step()
        return loss.detach(), loc_loss, regr_loss, theta_loss, accs.cpu().numpy()

    @torch.no_grad()
    def _validate_batch(self, inputs, model, criterion):
        """Validates the model on a single batch of data."""
        input, rois, rixs, clss, deltas, thetas = inputs
        model.eval()
        _clss, _theta_score, _deltas = model(input, rois, rixs)
        loss, loc_loss, regr_loss, theta_loss = criterion(_clss, _theta_score, _deltas, clss, thetas.view(-1, 1), deltas)
        _clss = self._decode(_clss)
        accs = clss == _clss
        return _clss, _deltas, loss.detach(), loc_loss, regr_loss, theta_loss, accs.cpu().numpy()

    def train_and_validate(self):
        """Runs the training and validation loops for multiple epochs."""
        if len(self.train_loader) > 0 and len(self.test_loader) > 0:
            frcnn = FRCNN().to(self.device)
            criterion = frcnn.calc_loss
            optimizer = SGD(frcnn.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
            log = Report(self.n_epochs)
            best_accuracy = 0

            for epoch in range(self.n_epochs):
                _n_train = len(self.train_loader)
                for ix, inputs in enumerate(self.train_loader):
                    loss, loc_loss, regr_loss, theta_loss, accs = self._train_batch(inputs, frcnn, optimizer, criterion)
                    pos = (epoch + (ix + 1) / _n_train)
                    self.writer.add_scalar("Loss/train", loss.item(), pos)
                    self.writer.add_scalar("Accuracy/train", accs.mean(), pos)
                    log.record(pos, trn_loss=loss.item(), trn_loc_loss=loc_loss, trn_regr_loss=regr_loss,
                               trn_theta_loss=theta_loss, trn_acc=accs.mean(), end='\r')

                _n_test = len(self.test_loader)
                val_losses = []

                for ix, inputs in enumerate(self.test_loader):
                    _clss, _deltas, loss, loc_loss, regr_loss, theta_loss, accs = self._validate_batch(inputs, frcnn, criterion)
                    pos = (epoch + (ix + 1) / _n_test)
                    val_losses.append(loss.item())
                    self.writer.add_scalar("Loss/val", loss.item(), pos)
                    self.writer.add_scalar("Accuracy/val", accs.mean(), pos)
                    log.record(pos, val_loss=loss.item(), val_loc_loss=loc_loss, val_regr_loss=regr_loss,
                               val_theta_loss=theta_loss, val_acc=accs.mean(), end='\r')

                    if loss.item() <= min(val_losses):
                        torch.save(frcnn.state_dict(), self.model_path)
                        best_accuracy = accs.mean()

                log.report_avgs(epoch + 1)

            self.writer.close()
            log.plot_epochs('trn_loss,val_loss'.split(','))

            with mlflow.start_run():
                mlflow.log_metric("accuracy", best_accuracy)
                mlflow.pytorch.log_state_dict(frcnn.state_dict(), 'model')
        else:
            print(f'Test loader: {len(self.test_loader)}, Train loader: {len(self.train_loader)}')



