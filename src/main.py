from typing import Callable
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from seg_model import SegmentationModel
from augmentor import Augmentor
from dataset import LandCoverDataset
from utils import Utils
from training_parameters import TrainingParameters
import segmentation_models_pytorch as smp

MODEL = "deeplabv3plus"

if MODEL == "unetplusplus":
    ENCODER =  "mobilenet_v2"
    PATH = "model_unet_7classes"
    DIR = "unet_7classes"
elif MODEL == "deeplabv3plus":
    ENCODER = "resnet50"
    PATH = "model_deepv3_7classes"
    DIR = "deepv3_7classes"

def main():
    train_split, val_split, test_split = read_data()
    rgb_values, class_names = get_class_info("class_dict.csv")
    indx = visualize(train_split, rgb_values)
    visualize_augmentations(train_split, rgb_values, indx)
    model, preprocessing_fn = create_model(ENCODER, 'imagenet', 7, MODEL)
    train_loader, valid_loader, test_loader = create_dataloaders(train_split, val_split, test_split, preprocessing_fn, rgb_values)
    params = define_parameters(model, n_classes=7, initial_lr= 1e-4 )
    train_logs, val_logs = model.train(params, train_loader, valid_loader, PATH)

    dump_logs(train_logs, val_logs, DIR)
    train_logs, val_logs = load_logs(DIR)
    visualize_training(train_logs, val_logs, DIR+"\\unet_7classes")

    model.load(PATH)

    f1, iou, loss = model.val_benchmark(valid_loader, rgb_values, class_names, params.loss, n_images = 10)
    print(f"Valid set Dice loss: {loss} | Valid set F1: {f1} | Valid set iou: {iou}")

    model.predict(test_loader, rgb_values, n_images = 5)

def benchmark():
    train_split, val_split, test_split = read_data()
    rgb_values, class_names = get_class_info("class_dict.csv")
    model, preprocessing_fn = create_model(ENCODER, 'imagenet', 7, MODEL)
    _, valid_loader, test_loader = create_dataloaders(train_split, val_split, test_split, preprocessing_fn, rgb_values)
    params = define_parameters(model, n_classes = 7, initial_lr = 1e-4)
    model.load(PATH)
    f1, iou, loss = model.val_benchmark(valid_loader, rgb_values, class_names, params.loss, DIR+"\\predictions2.png", n_images = 3)
    print(f"Valid set Dice loss: {loss} | Valid set F1: {f1} | Valid set iou: {iou}")

    # model.predict(test_loader, rgb_values, class_names, n_images = 2)
                     
def visualize_training(train_logs, val_logs, save_name):
    train_loss = train_logs["loss"][0]
    val_loss = val_logs["loss"][0]
    train_iou = train_logs["iou"][0]
    val_iou = val_logs["iou"][0]

    plt.figure(figsize = (12, 8))
    plt.plot(train_loss, label = "Train")
    plt.plot(val_loss, label = "Val")
    plt.legend()
    plt.title("Train vs Val loss")
    plt.savefig(save_name+"_loss.png")

    plt.show()

    plt.figure(figsize = (12, 8))
    plt.plot(train_iou, label = "Train")
    plt.plot(val_iou, label = "Val")
    plt.legend()
    plt.title("Train vs Val mean IoU")
    plt.savefig(save_name+"_iou.png")

    plt.show()

def read_data():
    metadata_df = pd.read_csv('metadata.csv')
    train_df = metadata_df[metadata_df['split']=='train']
    test_df = metadata_df[metadata_df['split']=='test']

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.drop('split', axis = 1)

    valid_df = train_df.sample(frac=0.1, random_state=42)
    train_df = train_df.drop(valid_df.index)

    print("Train", train_df.head())
    print("Train len: ", len(train_df))
    print("Val len: ", len(valid_df))
    return train_df, valid_df, test_df

def get_class_info(csv_path: str) -> tuple[np.array, ...]:
    class_dict = pd.read_csv(csv_path)
    
    class_dict.columns = class_dict.columns.str.strip()
    class_dict['name'] = class_dict['name'].str.strip()
    
    class_names = class_dict['name'].tolist()
    class_rgb_values = class_dict[['r', 'g', 'b']].values.astype(int).tolist()
    
    print('Class Names:', class_names)
    print('Class RGB values:', class_rgb_values)
    
    return np.array(class_rgb_values), np.array(class_names)

def visualize(dataset: pd.DataFrame, rgb_values: list) -> int:
    dataset = LandCoverDataset(dataset, class_rgb_values=rgb_values)
    random_idx = np.random.random_integers(0, len(dataset), 2)

    for i in random_idx:
        image, mask = dataset[i]
        mask = Utils.one_hot_encode(mask, rgb_values).astype('float')
        Utils.visualize(
            title="Original",
            original_image = image,
            ground_truth_mask = Utils.colour_code_segmentation(mask, rgb_values),
            one_hot_encoded_mask = mask
        )
    return random_idx

def visualize_augmentations(dataset: pd.DataFrame, rgb_values: list, random_idx: int):
    augmented_dataset = LandCoverDataset(
    dataset, 
    augmentation = Augmentor.get_training_augmentation(),
    class_rgb_values = rgb_values,
                )
    for i in random_idx:
        image, mask = augmented_dataset[i]
        Utils.visualize(
            title = "Augmented",
            original_image = image,
            ground_truth_mask = Utils.colour_code_segmentation(mask, rgb_values),
            one_hot_encoded_mask = mask
        )

def create_model(encoder: str, encoder_weights: str, n_classes: int, model_name: str) -> tuple[SegmentationModel, Callable]:
    model = SegmentationModel(encoder, model_name, encoder_weights, n_classes)
    preprocessing_fn = model.get_preprocessing()

    return model, preprocessing_fn

def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                        preprocessing_fn: Callable, rgb_values: list[float]) -> tuple[DataLoader, ...]:
    train_dataset = LandCoverDataset(
    train_df, 
    augmentation=Augmentor.get_training_augmentation(),
    preprocessing=Augmentor.get_preprocessing(preprocessing_fn),
    class_rgb_values=rgb_values,
    )

    valid_dataset = LandCoverDataset(
        val_df, 
        augmentation=Augmentor.get_validation_augmentation(),
        preprocessing=Augmentor.get_preprocessing(preprocessing_fn),
        class_rgb_values=rgb_values,
    )

    test_dataset = LandCoverDataset(
        test_df, 
        augmentation=Augmentor.get_validation_augmentation(),
        preprocessing=Augmentor.get_preprocessing(preprocessing_fn),
        class_rgb_values=rgb_values,
        test=True
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    return train_loader, valid_loader, test_loader

def define_parameters(model: SegmentationModel, n_classes: int, initial_lr: float) -> TrainingParameters:
    params = TrainingParameters
    params.epochs = 20
    params.warmup_epochs = 3
    params.loss = smp.losses.DiceLoss(classes = range(0, n_classes), mode = "multiclass", from_logits = False)
    optimizer = torch.optim.Adam([
        dict(params = model.get_params(), lr = initial_lr)
    ])
    params.optimizer  = optimizer

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max = params.epochs - params.warmup_epochs, eta_min = 6e-6
        )
    params.scheduler = lr_scheduler

    return params

def load_logs(DIR: str) -> tuple[dict, dict]:   
        with open(DIR + "\\train_logs.json", "r") as f:
            train_logs = json.load(f)

        with open(DIR+ "\\val_logs.json", "r") as f:
            val_logs = json.load(f)
        
        return train_logs, val_logs

def dump_logs(train_logs: dict, val_logs: dict, DIR: str) -> None:
        with open(DIR + "\\train_logs.json", "w") as f:
            json.dump(train_logs, f)

        with open(DIR+ "\\val_logs.json", "w") as f:
            json.dump(val_logs, f)

if __name__ == "__main__":
    # main()
    benchmark()