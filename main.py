import argparse
import sys, time

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type = str, default = "../imagenet")

    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 1e-3)

    parser.add_argument('--pruning_freq', type = int, default = 30,
                        help = 'do pruning for each frequency iterations')
    parser.add_argument('--max_pruned_neurons', type = int, default = 1500,
                        help = 'the number of maximum pruned gates')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args - parse_args()

    model = ResNet50(classifier_activation = 'softmax')
    #model = tf.keras.applications.resnet50.ResNet50(weights = 'imagenet', pooling = 'avg', classes = 1000, classifier_activation = 'softmax')
    candidate_parameters = select_pruning_parameters(model)
    gradient_pruning = GradientPruning(model, candidate_parameters)

    imgnet = load_dataset('imagenet-1k', cache_dir = args.data_dir)
    train_dataset, valid_dataset = imgnet['train'], imgnet['validation']

    train_loader = tf.data.Dataset.from_generator(generator,
                                                  output_shapes = ((224, 224, 3), ()),
                                                  output_types = (tf.float32, tf.int32)
                                                  )

    valid_loader = tf.data.Dataset.from_generator(valid_generator,
                                                  output_shapes = ((224, 224, 3), ()),
                                                  output_types = (tf.float32, tf.int32)
                                                  )

    optimizer = tf.keras.optimizers.experimental.SGD(learning_rate = args.lr, momentum = 0.9)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy()

    top1_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    start, end, average_inference = 0.0, 0.0, 0.0
    pruned_neurons, total_neurons = 0, 0
    n_trainable_vars = len(model.trainable_variables)

    for iter, data in enumerate(train_loader.batch(args.batch_size).prefetch(1)):

        images, labels = data
        images = preprocess_input(images)

        with tf.GradientTape() as g:
            preds = model(images)
            loss  = criterion(labels, preds)

        if (iter + 1) % args.pruning_freq == 0 and pruned_neurons <= args.max_pruned_neurons:
            gradients = g.gradient(loss, model.trainable_variables + candidate_parameters)

            train_gradients   = gradients[:n_trainable_vars]
            pruning_gradients = gradients[n_trainable_vars:]

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            pruned_neurons, total_neurons = gradient_pruning.step(pruning_gradients)
            print(f"[DEBUG] [{pruned_neurons} / {total_neurons}] are pruned")
        else:
            gradients = g.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        top1_metric.reset_state()
        top5_metric.reset_state()
        top1_metric.update_state(labels.numpy(), preds.numpy())
        top5_metric.update_state(labels.numpy(), preds.numpy())

        print(f"[{iter}]: Top1 {top1_metric.result().numpy():.5f} / Top5 {top5_metric.result().numpy():.5f} / pruning ratio {pruned_neurons/float(total_neurons) :.5f} sec")
