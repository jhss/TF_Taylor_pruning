import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
from functools import partial

import sys, time
from datasets import load_dataset

from models.resnet import ResNet50
from pruning_engine import GradientPruning, select_pruning_parameters

# load dataset and preprocessing

def generator():
    for data in train_dataset.shuffle():
        img, label = data['image'], data['label']
        img = (lambda x: x.convert('RGB') if x.mode != 'RGB' else x)(img)
        img = img.resize((224, 224))

        yield img, label

def valid_generator():
    for data in valid_dataset:
        img, label = data['image'], data['label']
        img = (lambda x: x.convert('RGB') if x.mode != 'RGB' else x)(img)
        img = img.resize((224, 224))

        yield img, label

model = ResNet50(classifier_activation = 'softmax')
#model = tf.keras.applications.resnet50.ResNet50(weights = 'imagenet', pooling = 'avg', classes = 1000, classifier_activation = 'softmax')
candidate_parameters = select_pruning_parameters(model)
gradient_pruning = GradientPruning(model, candidate_parameters)

imgnet = load_dataset('imagenet-1k')
train_dataset, valid_dataset = imgnet['train'], imgnet['validation']
num_train_data = len(train_dataset)
train_loader = tf.data.Dataset.from_generator(generator,
                                              output_shapes = ((224, 224, 3), ()),
                                              output_types = (tf.float32, tf.int32)
                                              )

valid_loader = tf.data.Dataset.from_generator(valid_generator,
                                              output_shapes = ((224, 224, 3), ()),
                                              output_types = (tf.float32, tf.int32)
                                              )

optimizer = tf.keras.optimizers.experimental.SGD(learning_rate = 0.001, momentum = 0.9)
criterion = tf.keras.losses.SparseCategoricalCrossentropy()

top1_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)

for data in train_loader.batch(128).prefetch(1):

    images, labels = data
    images = preprocess_input(images)

    with tf.GradientTape(persistent = True) as g:
        start = time.time()
        preds = model(images)
        end = time.time() - start
        print(f"[DEBUG] inference speed: {end:.5f} sec")
        loss  = criterion(labels, preds)

    gradients = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    pruning_gradients = g.gradient(loss, candidate_parameters)
    #print("[DEBUG] candidate parameter type: ", type(candidate_parameters[0]))
    gradient_pruning.step(pruning_gradients)
    #sys.exit()
    del g
    # del g, gradients, pruning_gradients (1200)

    #print("[DEBUG] type labels: ", type(labels.numpy()), " type preds: ", type(preds.numpy()))
    #print("preds.shape: ", preds.numpy().shape)
    #print("labels.shape: ", labels.numpy().shape)

    top1_metric.reset_state()
    top5_metric.reset_state()
    top1_metric.update_state(labels.numpy(), preds.numpy())
    top5_metric.update_state(labels.numpy(), preds.numpy())

    #prec1, prec5 = accuracy(preds, labels, topk = (1, 5))
    print(f"[DEBUG]: Top1 {top1_metric.result().numpy()} / Top5 {top5_metric.result().numpy()}")
