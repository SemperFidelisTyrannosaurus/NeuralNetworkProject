"""A convolutional neural network to classify between two possible classes of 
DNA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import dna_dataset

tf.logging.set_verbosity(tf.logging.INFO)

def calculate_confusion_matrix( results):   
    stats = {}
    cfmat = [[0, 0], [0, 0]];
    n = len( results)
    
    for r in results:
        cfmat[round(r[0])][round(r[1])] += 1

    stats["accuracy"] = ( cfmat[1][1] + cfmat[0][0]) / n
    stats["error"] = ( cfmat[1][0] + cfmat[0][1]) / n

    return (cfmat, stats)

def kmer_initializer(shape, dtype=None, partition_info=None):
  """This kernel initializer function sets the filters used in the first
  convolutional layer of the network to detect known features in the datasets
  to be classified."""
  kmers = dna_dataset.load_unique_kmers(shape[3], shape[1])
  return np.reshape(dna_dataset.kmer_seq_to_filters(kmers), shape)

def cnn_model_fn(features, labels, mode):
  k = 10

  # Define layers
  input_layer = tf.reshape(features["x"], [-1, 1, 100, 4])

  # Initial convolution layer uses filters defined from the 64 most common
  # kmer sequences from each class.
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=128,
      kernel_size=[1, k],
      padding="same",
      # kernel_initializer=kmer_initializer,
      activation=tf.nn.relu)

  # Initial pooling layer uses pools 10 bins at a time.
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 10], strides=10)

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[1, 5],
      padding="same",
      activation=tf.nn.relu)

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)
  pool2_flat = tf.reshape(pool2, [-1, 5 * 64])

  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=2)

  # Predictions
  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

  # Training
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Evaluation
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  """main operation loads the dataset, creates the estimator object using the 
  model function defined above, and begins training the dataset."""

  name = "10glorot"

  # Load the dataset.
  files = {
    "train" : ['HIV-1_train.txt','hg38_train.txt'],
    "test" : ['HIV-1_test.txt','hg38_test.txt']
  }
  train_data, train_labels = dna_dataset.load_labeled_data(files["train"])
  test_data, test_labels = dna_dataset.load_labeled_data(files["test"])

  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/dna_model_" + name)

  # Create a logger.
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Training
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=1000,
      num_epochs=None,
      shuffle=True)
  classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluation
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      shuffle=False)
  predict_results = classifier.predict(input_fn=predict_input_fn)

  results = [(test_labels[i],p['classes']) for i,p in enumerate(predict_results)]
  cfmat, stats = calculate_confusion_matrix( results)

  with open(name+"_results.md", "w") as resultsfile:
    resultsfile.write("### {}\n".format(name))
    resultsfile.write("|                | {:>14} | {:>14} |\n".format("actual hg38", "actual HIV1"))
    resultsfile.write("|----------------|----------------|----------------|\n")
    resultsfile.write("| {:<14} | {:>14} | {:>14} |\n".format("predicted hg38", cfmat[1][1], cfmat[1][0]))
    resultsfile.write("| {:<14} | {:>14} | {:>14} |\n".format("predicted HIV1", cfmat[0][1], cfmat[0][0]))
    resultsfile.write("\n")
    resultsfile.write("| {:<14} | {:>14} |\n".format("accuracy", "error"))
    resultsfile.write("|----------------|----------------|\n")
    resultsfile.write("| {:<14} | {:>14} |\n".format(stats["accuracy"], stats["error"]))

if __name__ == "__main__":
  tf.app.run()
