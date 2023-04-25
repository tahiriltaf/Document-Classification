
from tqdm import tqdm
import torch
from pathlib import Path
# from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import json
from PIL import Image, ImageDraw, ImageFont
from typing import List
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import easyocr
import os
import sys


# from torch import nn

# class ModelModule(nn.Module):

DOCUMENT_CLASSES = ['email', 'resume', 'scientific_publication']
pre_train_path = "/models/reference/microsoft-layoutlmv3-base"
# pre_train_path = "microsoft/layoutlmv3-base"
class ModelModule(pl.LightningModule):
    def __init__(self, n_classes:int):
        super().__init__()
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            pre_train_path, 
            num_labels=n_classes
        )
        self.model.config.id2label = {k: v for k, v in enumerate(DOCUMENT_CLASSES)}
        self.model.config.label2id = {v: k for k, v in enumerate(DOCUMENT_CLASSES)}
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        return self.model(
            input_ids, 
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bbox = batch["bbox"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, bbox, pixel_values, labels)
        self.log("train_loss", output.loss)
        self.log("train_acc", self.train_accuracy(output.logits, labels), on_step=True, on_epoch=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bbox = batch["bbox"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        output = self(input_ids, attention_mask, bbox, pixel_values, labels)
        self.log("val_loss", output.loss)
        self.log("val_acc", self.val_accuracy(output.logits, labels), on_step=False, on_epoch=True)
        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001) #1e-5
        return optimizer
    
model_module = ModelModule(len(DOCUMENT_CLASSES))
model_checkpoint = ModelCheckpoint(
    filename="{epoch}-{step}-{val_loss:.4f}", save_last=True, save_top_k=3, monitor="val_loss", mode="min"
)



def scale_bounding_box(box: List[int], width_scale : float = 1.0, height_scale : float = 1.0) -> List[int]:
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale)
    ]

def create_bounding_box(bbox_data):
    xs = []
    ys = []
    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)
    
    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))

    return [left, top, right, bottom]

def predict_document_image2(
    reader,
    image_path: Path, 
    model: LayoutLMv3ForSequenceClassification, 
    processor: LayoutLMv3Processor):

    with Image.open(image_path).convert("RGB") as image:

        width, height = image.size
        width_scale = 1000 / width
        height_scale = 1000 / height
        
        words = []
        boxes = []
        ocr_result = reader.readtext(str(image_path))    
        for bbox, word, confidence in ocr_result:
            boxes.append(scale_bounding_box(create_bounding_box(bbox), width_scale, height_scale))
            words.append(word)


        encoding = processor(
            image, 
            words,
            boxes=boxes,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    with torch.inference_mode():
        output = model(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
            bbox=encoding["bbox"].to(DEVICE),
            pixel_values=encoding["pixel_values"].to(DEVICE)
        )

    predicted_class = output.logits.argmax()
    return model.config.id2label[predicted_class.item()]

if __name__ == "__main__":
    easyocr_model = "/models/reference/EasyOCR/model"
    DOCUMENT_CLASSES = ['email', 'resume', 'scientific_publication']
    reader = easyocr.Reader(['en'],model_storage_directory=easyocr_model,download_enabled=False)

    model_path  = "/models/version_4/checkpoints/last.ckpt"
    trained_model = ModelModule.load_from_checkpoint(
        checkpoint_path = model_path, 
        n_classes=len(DOCUMENT_CLASSES), 
        local_files_only=True
    )
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = trained_model.model.eval().to(DEVICE)

    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(pre_train_path)
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)



    image_path = Path('/tst_images/tst_image.png')
    image_name = os.environ.get('IMAGE_NAME')
    if(image_name == 'None'):
        print('!!! image missing')
        print("docker run -e IMAGE_NAME=tst_image.png -v /absolute/path/to/images_dir:/images_dir dl_assignment_px_tahir")
    else:
        image_path = os.path.join('/images_dir', image_name)
    
        try:

            img = Image.open(image_path).convert("RGB")
            prediction = predict_document_image2(reader,image_path, model, processor)
            print(prediction)

            width, height = img.size
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            # font = ImageFont.truetype("arial.ttf", 50)
            text_width, text_height = draw.textsize(prediction, font)
            x = (width - text_width) // 2
            y = (height - text_height) // 2

            draw.text((x, y), prediction, font=font, fill=(255, 0, 0))
            img.save('/images_dir/tested.png')
            print("output saved in images_dir")
        # do something with the image
        except FileNotFoundError:
            print(f"Error: File '{image_path}' not found.", file=sys.stderr)
            print("!!! Give abolute path of folder in Host")
            print("docker run -e IMAGE_NAME=tst_image.png -v /absolute/path/to/images_dir:/images_dir dl_assignment_px_tahir")

    

 

    