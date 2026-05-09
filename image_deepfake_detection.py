# Importing the libraries and dependencies
import warnings
warnings.filterwarnings('ignore')  # Ignores the warning during execution
import gc # importing the gc module for garbage collection
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report,f1_score

from imblearn.over_sampling import RandomOverSampler
import accelerate
import evaluate
from datasets import Dataset, Image , ClassLabel
from transformers import Trainer, TrainingArguments, ViTImageProcessor,ViTForImageClassification,DefaultDataCollator

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop,Compose,Normalize,RandomRotation,RandomResizedCrop,RandomHorizontalFlip,RandomAdjustSharpness,Resize,ToTensor

from PIL import ImageFile
# Enable the option to load truncated images
# This setting allows the PIL library to attempt loading images even if they are corrupted or incomplete
ImageFile.LOAD_TRUNCATED_IMAGES = True

image_dict = {}

from pathlib import Path
from tqdm import tqdm
import os
import kagglehub
# Initializing empty lists to store lables and file names
file_names = []
labels = []
path = Path(__file__).parent / "Dataset"

# iterate through all image file in the specified dictionary
for file in Path(path).rglob("*.*"):
    if file.is_file():
        label = file.parent.name.lower()

        if label in ["fake", "real"]:
            labels.append(label)
            file_names.append(str(file))

# print(len(file_names),len(labels))

df = pd.DataFrame.from_dict({"Image" : file_names, "Label" : labels})
# print(df.shape)

# print(df.head())
# print(df['Label'].unique())

# Random over sampling of minority class to balance the dataset
# y contains the target variable (labels) that we want to predict
y = df['Label']

# Drop the label column from the dataset df to create the features from the target variable
df = df.drop('Label', axis=1)

# Create a RandomOverSampler object with a specified random seed
ros = RandomOverSampler(random_state=42)

# Use the oversampler to resample the dataset by oversampling the minority class
# The df contains the feature data and y_resampled contains the resampled target variable
df,y_resampled = ros.fit_resample(df,y)

# Delete the original 'y' variable to save the memory as it's no longer needed
del y

# Adding the resampled target variable as a new label column into the df
df['label'] = y_resampled
del y_resampled

# perform the garbage collection to free up space
gc.collect()
# print(df.shape)

# Creating a dataset from pandas dataframe and converting it to huggingface dataset

dataset = Dataset.from_pandas(df).cast_column("Image",Image())

labels_subset = labels[:5]
# print(labels_subset)

# Creating a list of unique labels by converting it to set then to list again
labels_list = ["real" , "fake"]

# Initializing empty dict to map labels to ids
label2id , id2label = dict() , dict()

for i, label in enumerate(labels_list):
    label2id[label] = i
    id2label[i] = label

# Creating a classlabels to match label and id
class_labels = ClassLabel(num_classes=len(labels_list) , names=labels_list)

# Mapping labels to ids
def map_label2id(batch):
    batch["label"] = [class_labels.str2int(x) for x in batch["label"]]
    return batch

dataset = dataset.map(map_label2id, batched=True)

# Casting label column to classLabel object
dataset = dataset.cast_column('label',class_labels)

# spliting the dataset into testing and training datasets
dataset = dataset.train_test_split(test_size=0.4,shuffle=True, stratify_by_column='label')

# Extracting the training and testing data from the split dataset
train_data = dataset['train']
test_data = dataset['test']

# Define the pre_trained ViT model string
model_str = "dima806/deepfake_vs_real_image_detection"

# create a processor for ViT model input
processor = ViTImageProcessor.from_pretrained(model_str)
image_mean,image_std = processor.image_mean , processor.image_std
size = processor.size["height"]
# print("size:" , size)

# Normalizing
normalize = Normalize(mean=image_mean, std=image_std)

# Defining a set of transformation for training data
_train_transform = Compose(
    [
        Resize((size,size)),
        RandomRotation(90),
        RandomAdjustSharpness(2),
        ToTensor(), # Convert images to tensors
        normalize
    ]
)

# Defining a set for validation data
_val_transform = Compose(
    [
        Resize((size,size)),
        ToTensor(),
        normalize
    ]
)

# Defining a funct to apply training transformation to a batch of examples
def train_transform(examples):
    examples['pixel_values'] = [_train_transform(image.convert('RGB')) for image in examples['Image']]
    return examples

# Defining a funct to apply val transformation to a batch of examples
def val_transform(examples):
    examples['pixel_values'] = [_val_transform(image.convert('RGB')) for image in examples['Image']]
    return examples

# set the transforms for training data
train_data.set_transform(train_transform)

# set the transforms for val/test data
test_data.set_transform(val_transform)

# defining a collate function that prepares batched data for model training 
def collate_fun(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values" : pixel_values , "labels":labels}


if __name__ == "__main__":
    # Loading, training and evaluating the model
    model = ViTForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))

    # FREEZE backbone for speed
    for param in model.vit.parameters():
        param.requires_grad = False

    model.config.id2label = id2label
    model.config.label2id = label2id
    # print(model.num_parameters(only_trainable=True) / 1e6) #trainable parameters in millions

    # Loading the accuracy metrics
    accuracy = evaluate.load('accuracy')

    # Defining a function to calculate evaluation metrics
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids
        predicted_labels = predictions.argmax(axis = 1)
        acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']
        return {"accuracy" : acc_score}

    # Defining the name ofo the evaluation metrics to be used during the training and evaluation
    metrics_name = "accuracy"
    model_name = "deepfake_vs_real_image_detection"
    train_epochs = 1

    # creating a instance of TrainingArguments to configure training settings
    args = TrainingArguments (
        output_dir=model_name,
        logging_dir='./logs',
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=train_epochs,
        weight_decay=0.02,    # To prevent Overfitting
        warmup_steps=50,
        remove_unused_columns=False,
        save_strategy='epoch',
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to='none',
        fp16=True,   #For GPU
        dataloader_num_workers=0
    )

    # Creating a Trainer instance for fine-tuning a language model
    trainer = Trainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fun,
        compute_metrics=compute_metrics,
        processing_class= processor
    )

    # Starting training the model using trainer object
    trainer.train()

    # Evaluating the model after training
    trainer.evaluate()

    # Prediction
    outputs = trainer.predict(test_data)
    # print(outputs.metrics)

    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    # Plotting the confusion matrix
    def plot_confusion_matrix(cm, classes, title = 'Confusion Matrix', cmap = plt.cm.Blues, figsize = (10,8)):
        plt.figure(figsize = figsize)
        plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
        plt.title(title)
        plt.colorbar()

        # Define tick marks and labels for the classes on the axes
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation = 90)
        plt.yticks(tick_marks, classes)

        fmt = '.0f'
        # Add text annotations to the plots indicating the values in the cells
        thresh = cm.max() / 2.0
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j,i,format(cm[i,j],fmt), horizontalalignment = 'center' , color = 'white' if cm[i,j] > thresh else 'black')

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()

    accuracy = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred , average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if len(labels_list) <= 150:
        cm = confusion_matrix(y_true,y_pred)
        plot_confusion_matrix(cm, labels_list, figsize=(8,6))

    print()
    print("Classification Report:")
    print()
    print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

    # saving the model
    trainer.save_model()