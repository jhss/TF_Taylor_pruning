import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from PIL import Image

import albumentations as A
from datasets import load_dataset

from models.resnet import ResNet50
# load dataset and preprocessing

"""
preprocessing = tf.keras.Sequential([
                    layers.Lambda(lambda x: x.convert('RGB') if x.mode !='RGB' else x),
                    layers.RandomCrop(224, 224),
                    layers.RandomFlip(mode = 'horizontal'),
                    layers.Normalization(mean = [0.485, 0.456, 0.406], variance = [0.229*0.229, 0.224*0.224, 0.225*0.225])
                ])
"""

def convert_rgb(image, **kwargs):
    
    if image.mode != 'RGB':
        return image.convert('RGB')
    else:
        return image

preprocessing = A.Compose([
                    #A.Lambda(image = convert_rgb, p = 1.0),
                    A.RandomResizedCrop(224, 224),
                    A.HorizontalFlip(),
                    A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), p =0.5)
                ])

def transform(examples):
    
    imgs = []
    labels = []

    print("[DEBUG] examples: ", examples)
    for example in examples:
        print("[DEBUG] example: ", example)
        img = Image.fromarray(example['image'])
        img = (lambda x: x.convert('RGB') if x.mode != 'RGB' else x)(img)

        imgs.append(preprocessing(image = np.array(img))['image'])
        labels.append(example['label'])
        
    #print("[DEBUG] tf.stack: ", imgs)
    imgs = tf.stack(imgs)
    print(labels[0].shape)
    print(labels)
    labels = tf.stack(labels)
    return imgs, labels

imgnet = load_dataset('imagenet-1k')
print("[DEBUG] imgnet: ", imgnet)
#train_dataset = imgnet['train'].to_tf_dataset(columns = ['image'], label_cols = ['label'], 
#                                              batch_size = 2, collate_fn = transform)
train_dataset = imgnet['train']
train_dataset.map(batch_size=32).set_transform(transform)
#print("[DEBUG] type: ", type(train_dataset))

for data in train_dataset:
    img, label = data['image'], data['label']
    print(img, label)
    break


#print(imgnet['train']['image'][0])
#tf_imgnet = imgnet.to_tf_dataset(
#resnet = ResNet50(weights='imagenet', pooling = 'avg', classifier_activation = 'softmax')
