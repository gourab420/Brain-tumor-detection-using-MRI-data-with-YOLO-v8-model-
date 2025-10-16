import os
import yaml
from os.path import exists
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import shutil
import glob
import logging
from tqdm import tqdm
from itertools import compress
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import cv2
import imutils
import ultralytics
from ultralytics import YOLO


pd.set_option("display.max_columns", None)
pd.set_option('display.max_colwidth', None)
plt.style.use("ggplot")
sns.set_palette(sns.diverging_palette(220, 20))
ultralytics.checks()

#dataset path
brain_tumor_dataset_directory = r"dataset\brain-tumor-dataset"

#create a dataFrame from image files in particular folder
def df_from_image_folders(images_path: str, extension: Optional[str] = "jpg") -> pd.DataFrame:
    label = []
    path = []
    image_files = glob.glob(os.path.join(images_path, "**", f"*.{extension.lower()}"), recursive=True)

    for file in image_files:
        dirpath = os.path.dirname(file)
        folder_name = os.path.basename(dirpath)
        label.append(folder_name)
        path.append(file)

    class_dict = {"path": path, "label": label}
    return pd.DataFrame(class_dict)

df = df_from_image_folders(brain_tumor_dataset_directory, extension = "jpg")
df.head()

df.shape

#display three images from the dataframe
def display_images(df: pd.DataFrame, img_width: int = 224) -> None:
    if len(df) < 3:
        print(f"the dataframe has {len(df)} rows, need at least 3 images.")
        return

    plt.figure(figsize=(15, 6))  
    
    # show three images 
    for i in range(3):
        image_path = df.iloc[i, 0]  
        img = cv2.imread(image_path)
        img = imutils.resize(img, width=img_width)
        plt.subplot(1, 3, i + 1)  
        plt.imshow(img)
        plt.axis('off')

    plt.show()

display_images(df)

#image processing
def process_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
        
        if image is None:
            return None  
        
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1  
        return width, height, channels  

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None  

def compute_image_statistics_from_df(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    grouped = df.groupby('label')

    for label, group in grouped:
        image_paths = group['path'].tolist()

        widths, heights, channel_counts = [], [], []

        # threadpoolexecutor for parallel execution
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_image, image_paths))

        results = [res for res in results if res is not None]

        if results:
            widths, heights, channel_counts = zip(*results)

            stats.append({
                'Fish Class': label,
                'Average Width': np.mean(widths),
                'Average Height': np.mean(heights),
                'Average Channels': np.mean(channel_counts),
                'Min Width': np.min(widths),
                'Max Width': np.max(widths),
                'Min Height': np.min(heights),
                'Max Height': np.max(heights)
            })

    return pd.DataFrame(stats)

df_statistics = compute_image_statistics_from_df(df)
df_statistics


def move_image(path_file: str, dest_folder: str) -> None:
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)

    try:
        shutil.move(path_file, dest_folder)
    except shutil.SameFileError:
        print("source and destination represent the same file.")

def check_image(source_folder: str, dest_folder: str, file: str, corrupted_files: list) -> None:
    path_folder = os.path.join(source_folder, file)
    
    if os.path.isdir(path_folder):
        for sub_file in os.listdir(path_folder):
            check_image(path_folder, dest_folder, sub_file, corrupted_files)
    else:
        if file.lower().endswith(('.jpg', '.png')):
            file_path = os.path.join(source_folder, file)
            img = cv2.imdecode(np.fromfile(file_path, np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"Corrupted image detected: {file_path}")
                move_image(file_path, dest_folder)
                corrupted_files.append(file_path)
              

def process_images(source_folder: str, dest_folder: str) -> None:
    corrupted_files = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for file in os.listdir(source_folder):
            futures.append(executor.submit(check_image, source_folder, dest_folder, file, corrupted_files))

        for future in tqdm(futures, desc="Processing images"):
            future.result()
    
    if corrupted_files:
        print(f"Corrupted images have been moved to the destination folder: {dest_folder}")
    else:
        print("No corrupted images found.")


source_folder = r"dataset\brain-tumor-dataset"
dest_folder = r"corrupt_images"

process_images(source_folder, dest_folder)


def get_difference_from_2_list(list1: List[int], list2: List[int]) -> List[int]:
    return list(set(list1) - set(list2))


def get_split_data(list_id: List[int], train_percentage: float, validation_percentage: float, test_percentage: float) -> Tuple[List[int], List[int], List[int]]:

    total = len(list_id)
    n_train = int((train_percentage / 100) * total)
    train = random.sample(list_id, n_train)

    list_id = get_difference_from_2_list(list_id, train)

    n_valid = int((validation_percentage / 100) * total)
    valid = random.sample(list_id, n_valid)

    test = get_difference_from_2_list(list_id, valid)

    return train, valid, test


def make_folders(destination_folder: str) -> None:
    folders = ["images", "labels"]
    inner_folders = ["train", "val", "test"]

    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)

    for folder in folders:
        path = os.path.join(destination_folder, folder)
        if not os.path.isdir(path):
            os.mkdir(path)

        for in_folder in inner_folders:
            inner_path = os.path.join(path, in_folder)
            if not os.path.isdir(inner_path):
                os.mkdir(inner_path)

#divided the dataset for training, validation and testing
def copy_image(file: str, source_folder: str, destination_folder: str, id_folder: int) -> None:

    inner_folders = ["train", "val", "test"]

    source = os.path.join(source_folder, file)
    destination = os.path.join(destination_folder, 'images', inner_folders[id_folder], file)
    try:
        shutil.copy(source, destination)
    except shutil.SameFileError:
        print(f"Source and destination represent the same file: {file}")

    separator = file.find(".")
    filename = file[:separator] + ".txt"
    source = os.path.join(source_folder, filename)
    destination = os.path.join(destination_folder, 'labels', inner_folders[id_folder], filename)
    try:
        shutil.copy(source, destination)
    except shutil.SameFileError:
        print(f"Source and destination represent the same file: {filename}")


def copy_files_parallel(files: List[str], source_folder: str, destination_folder: str, id_folder: int) -> None:
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(copy_image, file, source_folder, destination_folder, id_folder) for file in files]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Copying files to {['train', 'val', 'test'][id_folder]}"):
            future.result()  


def split_and_copy_dataset(source_folder: str, destination_folder: str, train_percentage: float, validation_percentage: float, test_percentage: float) -> pd.DataFrame:
    
    if train_percentage < validation_percentage or train_percentage < test_percentage:
        print("Train set must have the biggest percentage.")
        return pd.DataFrame()

    total_percentage = train_percentage + validation_percentage + test_percentage
    if total_percentage != 100:
        print("Total percentage must be 100%.")
        return pd.DataFrame()

    list_id = [count for count, file in enumerate(os.listdir(source_folder)) if file.endswith((".jpg", ".png"))]
    train, valid, test = get_split_data(list_id, train_percentage, validation_percentage, test_percentage)
    make_folders(destination_folder)
    train_files = [file for count, file in enumerate(os.listdir(source_folder)) if count in train]
    valid_files = [file for count, file in enumerate(os.listdir(source_folder)) if count in valid]
    test_files = [file for count, file in enumerate(os.listdir(source_folder)) if count in test]

    print("Copying train files...")
    copy_files_parallel(train_files, source_folder, destination_folder, 0)
    print("Copying validation files...")
    copy_files_parallel(valid_files, source_folder, destination_folder, 1)
    print("Copying test files...")
    copy_files_parallel(test_files, source_folder, destination_folder, 2)

    data = {
        "Set": ["Train", "Validation", "Test"],
        "Images Count": [len(train_files), len(valid_files), len(test_files)],
        "Labels Count": [len(train_files), len(valid_files), len(test_files)]
    }
    df = pd.DataFrame(data)

    print("Dataset splitting and copying completed.")
    return df


source_folder = r"dataset\brain-tumor-dataset"
destination = r"brain-tumor-dataset-splitted"

df = split_and_copy_dataset(source_folder, destination, 80, 10, 10)
df.head()

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch version: {torch.__version__}")


config_data = {
    "path": r"brain-tumor-dataset-splitted",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": ["Tumor"]
}

with open('config.yaml', 'w') as f:
    yaml.dump(config_data, f, default_flow_style=None, sort_keys=False)

data_path = 'config.yaml'

model = YOLO("yolov8n.pt")
model.train(
    data=data_path,
    epochs=100,
    patience=10,
    batch=32,            
    imgsz=640,
    device=0,
    optimizer="AdamW",
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=3,
    save=True,
    cache="disk",          
    amp=True,
    workers=0,             
    resume=False,
    project="brain_tumor_project",
    name="yolov8_object_detection",
)
