#!/usr/bin/env python
# coding: utf-8


from cProfile import label
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import copy
import json
import shutil
from PIL import Image
import time
from pathlib import Path
from random import random, seed
from typing import List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset, load_metric
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from captioned_image import CaptionedImage
from tfidf import get_tfidf_vector, train_tfidf

DATA_BASE_PATH = Path("/local/s3052249")
BIRDS_PATH = DATA_BASE_PATH / "data/birds/birds.json"
FLOWERS_PATH = DATA_BASE_PATH / "data/flowers/flowers.json"
EMBEDDINGS_BASE_PATH = DATA_BASE_PATH / "embeddings"
TEMP_PATH = DATA_BASE_PATH / "temp"
EMBEDDINGS_BASE_PATH.mkdir(parents=True, exist_ok=True)
TEMP_PATH.mkdir(parents=True, exist_ok=True)
IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 32
IMAGE_FINETUNING_EPOCHS = 20
TEXT_FINETUNING_EPOCHS = 20
TRAIN_TEST_SPLIT = 0.7
IMAGE_VALIDATION_SPLIT = 0.8
IMAGE_LEARNING_RATE = 0.001

plt.rcParams["figure.figsize"] = (30, 15)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["font.size"] = 32
plt.rcParams["axes.xmargin"] = 0

EMBEDDINGS_BASE_PATH.mkdir(exist_ok=True, parents=True)

seed(42) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


with open(FLOWERS_PATH) as f:
    flowers = [CaptionedImage.parse_obj(v) for v in json.load(f)]

with open(BIRDS_PATH) as f:
    birds = [CaptionedImage.parse_obj(v) for v in json.load(f)]


bird_classes = list(set(b.class_name for b in birds))
flower_classes = list(set(f.class_name for f in flowers))

bird_test_classes = bird_classes[round(len(bird_classes) * TRAIN_TEST_SPLIT) :]
bird_train_classes = [c for c in bird_classes if c not in bird_test_classes]

flower_test_classes = flower_classes[round(len(flower_classes) * TRAIN_TEST_SPLIT) :]
flower_train_classes = [c for c in flower_classes if c not in flower_test_classes]

print(
    len(bird_test_classes),
    len(bird_train_classes),
    len(flower_test_classes),
    len(flower_train_classes),
)

train_birds = [b for b in birds if b.class_name in bird_train_classes]
test_birds = [b for b in birds if b.class_name in bird_test_classes]
train_flowers = [f for f in flowers if f.class_name in flower_train_classes]
test_flowers = [f for f in flowers if f.class_name in flower_test_classes]


def train_model(
    model, dataloaders, criterion, optimizer, num_epochs, is_inception=False
):
    since = time.time()

    train_acc_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == "train":
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
            else:
                train_acc_history.append(epoch_acc)


        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, list(zip(train_acc_history, val_acc_history))


def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """Resnet18"""
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """VGG11_bn"""
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    model_ft.to(device)
    return model_ft, input_size


def finetune_image_embedding(
    classes: List[str], data: List[CaptionedImage], model_name
):
    shutil.rmtree(TEMP_PATH)

    for c in classes:
        train_folder = TEMP_PATH / "train" / c
        train_folder.mkdir(parents=True, exist_ok=True)
        validation_folder = TEMP_PATH / "val" / c
        validation_folder.mkdir(parents=True, exist_ok=True)
        for d in data:
            if d.class_name == c:
                path = (DATA_BASE_PATH / d.image_path).with_suffix('.png')
                image = Image.open(path)
                if image.mode != 'RGB':
                    print(f'bad format ({image.mode}) {path}')
                    continue
                if random() < IMAGE_VALIDATION_SPLIT:
                    shutil.copy(path, train_folder)
                else:
                    shutil.copy(path, validation_folder)

    num_classes = len(classes)
    model_ft, input_size = initialize_model(
        model_name, num_classes, use_pretrained=True
    )

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(TEMP_PATH / x, data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=IMAGE_BATCH_SIZE, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }

    params_to_update = model_ft.parameters()

    optimizer_ft = optim.SGD(params_to_update, lr=IMAGE_LEARNING_RATE, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    model_ft, hist = train_model(
        model_ft,
        dataloaders_dict,
        criterion,
        optimizer_ft,
        num_epochs=IMAGE_FINETUNING_EPOCHS,
        is_inception=(model_name == "inception"),
    )

    if model_name == "vgg":
        del model_ft.classifier[6]
    else:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Identity(num_ftrs)

    plt.title(f"Validation accuracy of {model_name.upper()}")
    plt.xlabel("Epochs", labelpad=20)
    plt.ylabel("Accuracy", labelpad=20)
    plt.plot(range(1, IMAGE_FINETUNING_EPOCHS + 1), [t.cpu().numpy() for t, v in hist], '--', label=f'Train - {model_name.upper()}')
    plt.plot(range(1, IMAGE_FINETUNING_EPOCHS + 1), [v.cpu().numpy() for t, v in hist], label=f'Validation - {model_name.upper()}')
    plt.legend()
    plt.ylim((0, 1.0))
    plt.savefig(f"{model_name}-{classes[0]}.png")

    return model_ft, input_size


def create_image_embedding(
    classes: List[str], data: List[CaptionedImage], model, input_size
):
    shutil.rmtree(TEMP_PATH)

    paths = []
    for c in classes:
        test_folder = TEMP_PATH / "test" / c
        test_folder.mkdir(parents=True, exist_ok=True)
        for d in data:
            if d.class_name == c:
                path = (DATA_BASE_PATH / d.image_path).with_suffix('.png')
                image = Image.open(path)
                if image.mode != 'RGB':
                    print(f'bad format ({image.mode}) {path}')
                    continue
                shutil.copy(path, test_folder)
                paths.append(d.image_path)

    image_dataset = datasets.ImageFolder(
        TEMP_PATH / "test",
        transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False, num_workers=4
    )
    result = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            result.extend(outputs)

    return {p: v for p, v in zip(paths, result)}


def generate_image_embeddings(
    train_classes, test_classes, data: List[CaptionedImage], name: str
) -> None:
    finetuned_model, input_size = finetune_image_embedding(train_classes, data, "vgg")
    vgg_embeddings_train = create_image_embedding(
        train_classes, data, finetuned_model, input_size
    )
    vgg_embeddings_test = create_image_embedding(
        test_classes, data, finetuned_model, input_size
    )
    del finetuned_model

    finetuned_model, input_size = finetune_image_embedding(
        train_classes, data, "inception"
    )
    inception_embeddings_train = create_image_embedding(
        train_classes, data, finetuned_model, input_size
    )
    inception_embeddings_test = create_image_embedding(
        test_classes, data, finetuned_model, input_size
    )
    del finetuned_model

    finetuned_model, input_size = finetune_image_embedding(
        train_classes, data, "resnet"
    )
    resnet_embeddings_train = create_image_embedding(
        train_classes, data, finetuned_model, input_size
    )
    resnet_embeddings_test = create_image_embedding(
        test_classes, data, finetuned_model, input_size
    )
    del finetuned_model

    joblib.dump(
        {
            d.image_path: {
                "class_name": d.class_name,
                "vgg": vgg_embeddings_train[d.image_path],
                "resnet": resnet_embeddings_train[d.image_path],
                "inception": inception_embeddings_train[d.image_path],
            }
            for d in data
            if d.image_path in vgg_embeddings_train
        },
        EMBEDDINGS_BASE_PATH / f"image_embeddings_train_{name}.p",
    )

    joblib.dump(
        {
            d.image_path: {
                "class_name": d.class_name,
                "vgg": vgg_embeddings_test[d.image_path],
                "resnet": resnet_embeddings_test[d.image_path],
                "inception": inception_embeddings_test[d.image_path],
            }
            for d in data
            if d.image_path in vgg_embeddings_test
        },
        EMBEDDINGS_BASE_PATH / f"image_embeddings_test_{name}.p",
    )


def finetune_bert(classes: List[str], data):
    captions = []
    for d in data:
        for caption in d.captions:
            captions.append(
                {
                    "text": caption,
                    "label": classes.index(d.class_name),
                }
            )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataset = (
        Dataset.from_pandas(pd.DataFrame(captions))
        .map(lambda v: tokenizer(v["text"], truncation=True), batched=True)
        .train_test_split(test_size=0.2, shuffle=True)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(classes)
    )
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=DATA_BASE_PATH / "results",
        learning_rate=2e-5,
        per_device_train_batch_size=TEXT_BATCH_SIZE,
        per_device_eval_batch_size=TEXT_BATCH_SIZE,
        num_train_epochs=TEXT_FINETUNING_EPOCHS,
        evaluation_strategy="epoch",
        weight_decay=0.01,
    )

    Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    ).train()

    return model, tokenizer


def create_text_embedding(model, tokenizer, text):
    tensors = tokenizer(text, return_tensors="pt", padding=True)
    for k, t in tensors.items():
        tensors[k] = t.cuda()

    v = model.distilbert(**tensors)

    return v.last_hidden_state[0][0].cpu().detach().numpy()


def generate_text_embeddings(
    train_classes, test_classes, data: List[CaptionedImage], name: str
) -> None:
    train_data = [d for d in data if d.class_name in train_classes]
    test_data = [d for d in data if d.class_name in test_classes]

    tfidf_model = train_tfidf(train_data)

    bert, tokenizer = finetune_bert(train_classes, train_data)

    joblib.dump(
        {
            d.image_path: {
                text: {
                    "class_name": d.class_name,
                    "bert": create_text_embedding(bert, tokenizer, text),
                    "tfidf": get_tfidf_vector(tfidf_model, text),
                }
                for text in d.captions
            }
            for d in tqdm(train_data)
        },
        EMBEDDINGS_BASE_PATH / f"text_embeddings_train_{name}.p",
    )

    joblib.dump(
        {
            d.image_path: {
                text: {
                    "class_name": d.class_name,
                    "bert": create_text_embedding(bert, tokenizer, text),
                    "tfidf": get_tfidf_vector(tfidf_model, text),
                }
                for text in d.captions
            }
            for d in tqdm(test_data)
        },
        EMBEDDINGS_BASE_PATH / f"text_embeddings_test_{name}.p",
    )

    return tfidf_model


def generate_embeddings(
    train_classes, test_classes, data: List[CaptionedImage], name: str
) -> None:
    generate_image_embeddings(train_classes, test_classes, data, name)
    generate_text_embeddings(train_classes, test_classes, data, name)

if __name__ == '__main__':
    generate_embeddings(bird_train_classes, bird_test_classes, birds, "birds")
    plt.clf()
    generate_embeddings(flower_train_classes, flower_test_classes, flowers, "flowers")
