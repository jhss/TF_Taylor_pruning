import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from PIL import Image
from functools import partial

import albumentations as A
from datasets import load_dataset

from models.resnet import ResNet50
# load dataset and preprocessing

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


def generator():
    for data in train_dataset.shuffle():
        img, label = data['image'], data['label']
        #print("[DEBUG] generator type img: ", type(img))
        img = (lambda x: x.convert('RGB') if x.mode != 'RGB' else x)(img)
        img = preprocessing(image = np.array(img))['image']

        yield img, label

model = ResNet50(classifier_activation = 'softmax')

imgnet = load_dataset('imagenet-1k')
train_dataset = imgnet['train']
num_train_data = len(train_dataset)
train_loader = tf.data.Dataset.from_generator(generator, 
                                              output_shapes = ((224, 224, 3), ()),
                                              output_types = (tf.float32, tf.int32)
                                              )

optimizer = tf.keras.optimizers.experimental.SGD(learning_rate = 0.001, momentum = 0.9, weight_decay = 0.0)
criterion = tf.keras.losses.SparseCategoricalCrossentropy()

top1_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

for data in train_loader.batch(256).prefetch(1):

    images, labels = data

    with tf.GradientTape() as g:
        preds = model(images)
        loss  = criterion(labels, preds)

    gradients = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print("[DEBUG] type labels: ", type(labels.numpy()), " type preds: ", type(preds.numpy()))
    print("preds.shape: ", preds.numpy().shape)
    print("labels.shape: ", labels.numpy().shape)
    
    top1_metric.reset_state()
    top5_metric.reset_state()
    top1_metric.update_state(labels.numpy(), preds.numpy())
    top5_metric.update_state(labels.numpy(), preds.numpy())

    #prec1, prec5 = accuracy(preds, labels, topk = (1, 5))
    print(f"[DEBUG]: Top1 {top1_metric.result().numpy()} / Top5 {top5_metric.result().numpy()}")


    

