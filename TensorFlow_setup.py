# based on this guide: https://www.tensorflow.org/get_started/eager

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

import datetime
import pandas as pd
import numpy as np

# load in the data
# replace by owrn own parsing

metadata_file = '/Users/jaspermeijering/Google Drive/a Study/EPA Study Abroad - Carnegie Mellon University/Courses/CMU - 95845 - Applied Analytics The Machine Learning Pipeline/Machine Learning Pipeline Final Project/Data/FalseNews_Code_Data/data/metadata_anon.txt'

# Read meta data
fin = open(metadata_file, 'r')
lines = fin.readlines()
fin.close()
cascade_id2metadata = {}
for line in lines:
    line = line.replace('\n', '')
    item = eval(line)
    cascade_id2metadata[item[0]] = item[1]

    # Get static measures
veracity = []
virality = []
depth = []
breadth = []
size = []
verified = []
nfollowers = []
nfollowees = []
engagement = []
category = []
day = []
hour = []

for cascade, metadata in cascade_id2metadata.items():
    veracity.append(metadata['veracity'])
    virality.append(metadata['virality'])
    depth.append(metadata['depth'])
    breadth.append(metadata['max_breadth'])
    size.append(metadata['size'])
    verified.append(metadata['verified_list'][0])
    nfollowers.append(metadata['num_followers_list'][0])
    nfollowees.append(metadata['num_followees_list'][0])
    engagement.append(metadata['engagement_list'][0])
    category.append(metadata['rumor_category'])
    day.append(metadata['start_date'].day)
    hour.append(metadata['start_date'].hour)

# Convert to data frame
df = pd.DataFrame({'veracity': veracity, 'virality': virality, 'depth': depth, 'breadth': breadth, 'size': size, 'verified': verified, 'nfollowers': nfollowers,
                   'nfollowees': nfollowees, 'engangement': engagement, 'category': category, 'day': day, 'hour': hour})

# Inspect
df.head(5)


def parse_data(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
    parsed_line = tf.decode_csv(line, example_defaults)
    # First 4 fields are features, combine into single tensor
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    # Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label


for i in range(len(df))
random_num = random.random()
    df[i, train_id] = ifelse(random_num < 0.5, 0, 1)

train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)             # skip the first header row
train_dataset = train_dataset.map(parse_data)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(32)


features = dataset.iloc[:, :].values
labels = dataset.iloc[:, 0].values

# View a single example entry from a batch
features, label = tfe.Iterator(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])

# create a model using keras. Here in this case, two Dense layers with 10 nodes each, and an output layer with 2 nodes representing our label predictions.
# here the activation fuction is set to "relu"
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(2)
])

# Define the loss and gradient function 2 methods

# Method 1 from tutorial


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)

# With mean_squared_error : see: https://www.tensorflow.org/api_docs/python/tf/losses/mean_squared_error


tf.losses.mean_squared_error(
    labels,
    predictions,
    weights=1.0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)

# Create an optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# Training loop
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 32
    for x, y in tfe.Iterator(train_dataset):
        # Optimize the model
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, x, y))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

# Visualize the loss function over time

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()

# load in the test date
# replace by owrn own
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

test_dataset = tf.data.TextLineDataset(test_fp)
test_dataset = test_dataset.skip(1)             # skip header row
test_dataset = test_dataset.map(parse_csv)      # parse each row with the funcition created earlier
test_dataset = test_dataset.shuffle(1000)       # randomize
test_dataset = test_dataset.batch(32)

# Evaluate the model on the test dataset

test_accuracy = tfe.metrics.Accuracy()

for (x, y) in tfe.Iterator(test_dataset):
    prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# Use the trained model to make predictions
class_ids = ["True news", "Fake news"]  # or other way around?
predict_dataset = tf.convert_to_tensor([

])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    name = class_ids[class_idx]
    print("Example {} prediction: {}".format(i, name))

# Where to set up regularization?
