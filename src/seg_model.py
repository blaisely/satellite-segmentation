from tqdm import tqdm
from typing import Iterator, Callable

from torch.amp import GradScaler
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from training_parameters import TrainingParameters
from utils import Utils

class EarlyStopper:
    def __init__(self, patience: int = 4, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class SegmentationModel:
    def __init__(self, encoder: str, model_name: str, encoder_weights: str, n_classes: int):
        """
        Docstring for __init__
        
        :param encoder: Name of the used encoder
        :type encoder: str
        :param model_name: Name of the model: 
         - "deeplabv3plus" - for DeepLabV3Plus
         - "unetplusplus" - for Unet++
        :type model_name: str
        :param encoder_weights: Pretrained model weights type
        :type encoder_weights: str
        :param n_classes: Number of classes
        :type n_classes: int
        """
        self._model_name = model_name
        self._encoder = encoder
        self._weights = encoder_weights
        self._n_classes = n_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        if model_name == "deeplabv3plus":
            self._model = smp.DeepLabV3Plus(
            encoder_name=encoder, 
            encoder_weights=encoder_weights, 
            classes=n_classes, 
            activation="softmax",
            )

        elif model_name == "unetplusplus":
            self._model = smp.UnetPlusPlus(
            encoder_name = encoder, 
            encoder_weights = encoder_weights, 
            classes = n_classes, 
            activation = "softmax"
            )

    def get_model(self):
        return self._model
    
    def get_params(self) -> Iterator[torch.nn.Parameter]:
        return self._model.parameters()
    
    def get_preprocessing(self) -> callable:
        return smp.encoders.get_preprocessing_fn(encoder_name = self._encoder,
                                                 pretrained=self._weights)
    
    def load(self, path: str) -> None:
        self._model = smp.from_pretrained(path)

    def train(self, training_params: TrainingParameters, train_dataset: DataLoader, val_dataset: DataLoader, save_path: str) -> tuple[dict, dict]:
        optmizer = training_params.optimizer
        epochs = training_params.epochs
        loss = training_params.loss
        scheduler = training_params.scheduler
        warmup_epochs = training_params.warmup_epochs

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optmizer, 
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
        )

        scheduler_combined = torch.optim.lr_scheduler.SequentialLR(
        optmizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[warmup_epochs]
        )

        best_iou_score = 0.0
        train_logs_list, valid_logs_list = {"loss": [], "iou": [], "f1": []}, {"loss": [], "iou": [], "f1": []}
        running_loss_train, running_loss_val = [], []
        runnning_iou_train, running_iou_val = [], []
        running_f1_train, running_f1_val = [], []

        early_stopper = EarlyStopper(4, 0.001)
        for epoch in tqdm(range(epochs), desc = "Epochs"):
            
            train_loss, train_iou, train_f1 = self.run_epoch(train_dataset, optmizer, loss)
            val_loss, val_iou, val_f1 = self.run_validation(val_dataset, loss)

            scheduler_combined.step()

            running_loss_train.append(train_loss)
            runnning_iou_train.append(train_iou)
            running_f1_train.append(train_f1)
            running_loss_val.append(val_loss)
            running_iou_val.append(val_iou)
            running_f1_val.append(val_f1)

            if early_stopper.early_stop(val_loss):             
                break

            if best_iou_score < val_iou:
                best_iou_score = val_iou
                self._model.save_pretrained(save_path)
                print('Model saved!')
            
            print(f'Epoch: {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.5f} | Train IoU: {train_iou:.5f} | Train F1 {train_f1:.5f}')
            print(f'Val Loss: {val_loss:.5f} | Val IoU: {val_iou:.5f} | Val F1 {val_f1:.5f}')

        train_logs_list["loss"].append(running_loss_train)
        train_logs_list["iou"].append(runnning_iou_train)
        train_logs_list["f1"].append(running_f1_train)

        valid_logs_list["loss"].append(running_loss_val)
        valid_logs_list["iou"].append(running_iou_val)
        valid_logs_list["f1"].append(running_f1_val)
      
        return train_logs_list, valid_logs_list

    def run_epoch(self, loader: DataLoader, optimizer: torch.optim, loss_fn: Callable) -> tuple[dict, dict, dict]:
        train_loss, train_iou, train_f1 = 0, 0, 0
        self._model.train()
        self._model = self._model.to(self.device)
        num_batches = len(loader)

        scaler = GradScaler()
        for X_batch, y_batch in tqdm(loader, desc="Batch"):
            X_batch, y_batch = X_batch.to(self.device).float(), y_batch.to(self.device).long()
            
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda"):
                output = self._model(X_batch)
                loss = loss_fn(output, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm = 1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                preds = torch.argmax(output, dim = 1)
                new_iou, new_f1 = self._calculate_stats(preds, y_batch)

            train_loss += loss.item()
            train_iou += new_iou.item()
            train_f1 += new_f1.item()

        return train_loss / num_batches, train_iou / num_batches, train_f1 / num_batches
    
    def run_validation(self, loader: DataLoader, loss_fn: Callable) -> tuple[dict, dict, dict]:
        val_loss, val_iou, val_f1 = 0, 0, 0
        self._model.eval()

        num_batches = len(loader)
        for X_batch, y_batch in tqdm(loader, desc = "Valid"):
            X_batch, y_batch = X_batch.to(self.device).float(), y_batch.to(self.device).long()

            with torch.no_grad():
                output = self._model(X_batch)
                preds = torch.argmax(output, dim = 1)
                loss = loss_fn(output, y_batch)
            
            val_loss += loss.item()
            new_iou, new_f1 = self._calculate_stats(preds, y_batch)
            val_iou += new_iou.item()
            val_f1 += new_f1.item()
        
        return val_loss / num_batches, val_iou / num_batches, val_f1 / num_batches

    def val_benchmark(self, val_loader: DataLoader, class_rgb_values: list[float], class_names: list[str],
                       loss_fn: Callable, save_path: str, n_images: int = 5 ) -> tuple[dict, dict, dict]:
        self._model.eval()
        self._model.to(self.device)
        running_iou, running_f1, running_loss = 0, 0, 0
        images_shown = 0
        images, masks, preds_list = [], [], []

        for X_batch, y_batch in tqdm(val_loader, desc="Benchmarking"):
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).long()
            
            with torch.no_grad():
                output = self._model(X_batch)
                preds = torch.argmax(output, dim=1)
                loss = loss_fn(output, y_batch)
                
            iou, f1 = self._calculate_stats(preds, y_batch)
            running_iou += iou if isinstance(iou, float) else iou.item()
            running_f1 += f1 if isinstance(f1, float) else f1.item()
            running_loss += loss if isinstance(loss, float) else loss.item()

            if images_shown < n_images:
                for i in range(preds.shape[0]):
                    if images_shown >= n_images: break
                    
                    image = X_batch[i].cpu().permute(1, 2, 0).numpy()
                    image = Utils.denormalize(image)
                    mask = y_batch[i].cpu().numpy()
                    mask_pred = preds[i].cpu().numpy()

                    images.append(image)
                    masks.append(mask)
                    preds_list.append(mask_pred)
                    images_shown += 1

        Utils.visualize_batch(
            images = images,
            masks = masks,
            preds = preds_list,
            class_rgb_values = class_rgb_values,
            class_names = class_names,
            title="Batch Predictions",
            save=True,
            save_path=save_path
        )

        return running_f1 / len(val_loader), running_iou / len(val_loader), running_loss / len(val_loader)
    
    def predict(self, test_loader: DataLoader, class_rgb_values: list[float],
                 class_names: list[str], visualize: bool = True, n_images: int = 5) -> None:
        self._model.eval()
        self._model.to(self.device)
        images_shown = 0
        for X_batch in tqdm(test_loader, desc="Predicting"):
            X_batch = X_batch.to(self.device).float()

            with torch.no_grad():
                output = self._model(X_batch)
                preds = torch.argmax(output, dim=1).cpu().numpy()
            
            if visualize and images_shown < n_images:
                for i in range(preds.shape[0]):
                    if images_shown >= n_images:
                        break
                    image = X_batch[i].cpu().permute(1, 2, 0).numpy()
                    image = Utils.denormalize(image)
                    preds = preds[i]
                    Utils.visualize(
                        title = "Prediction",
                        class_rgb_values = class_rgb_values,
                        class_names = class_names,
                        Original_image = image,
                        Prediction = Utils.colour_code_segmentation(preds, class_rgb_values),
                        )
                    images_shown += 1
    
    def _calculate_stats(self, preds: torch.Tensor, y_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tp, fp, fn, tn = smp.metrics.get_stats(preds, y_batch, num_classes = self._n_classes, mode = "multiclass")
        train_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        train_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        return train_iou, train_f1
    
    def save(self, path: str) -> None:
        self._model.save_pretrained(save_directory=path)
