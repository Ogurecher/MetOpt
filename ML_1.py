import pandas
import numpy as np
import tensorflow
import math
from tensorflow.python.data import Dataset
from sklearn import metrics

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

def rmsleCalc(targets, predictions):
    sum=0.0
    for x in range(len(predictions)):
        if predictions[x]<0 or targets[x]<0: #check for negative values
            continue
        p = np.log(predictions[x]+1)
        r = np.log(targets[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predictions))**0.5

dataframe = pandas.read_csv("train.csv")
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))

features = dataframe[["GrLivArea"]]
feature_columns = [tensorflow.feature_column.numeric_column("GrLivArea")]

targets = dataframe["SalePrice"]

my_optimizer=tensorflow.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tensorflow.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tensorflow.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(features, targets),
    steps=100
)

prediction_input_fn =lambda: my_input_fn(features, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a np array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
rmsle = rmsleCalc(targets, predictions)

def train_model(learning_rate, steps, batch_size, input_feature="GrLivArea"):
  """Trains a linear regression model of one feature.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `dataframe`
      to use as input feature.
  """

  periods = 10
  steps_per_period = steps / periods

  features = input_feature
  features_data = dataframe[[features]]
  my_label = "SalePrice"
  targets = dataframe[my_label]

  # Create feature columns.
  feature_columns = [tensorflow.feature_column.numeric_column(features)]

  # Create input functions.
  training_input_fn = lambda:my_input_fn(features_data, targets, batch_size=batch_size)
  prediction_input_fn = lambda: my_input_fn(features_data, targets, num_epochs=1, shuffle=False)

  # Create a linear regressor object.
  my_optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tensorflow.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tensorflow.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )

  # Set up to plot the state of our model's line each period.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(my_label)
  plt.xlabel(features)
  sample = dataframe.sample(n=300)
  plt.scatter(sample[features], sample[my_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSLE (on training data):")
  rmsleArr = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])

    # Compute loss.
    rmsle = rmsleCalc(targets, predictions)
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, rmsle))
    # Add the loss metrics from this period to our list.
    rmsleArr.append(rmsle)
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, sample[my_label].max()])

    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[features].max()),
                           sample[features].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period])
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSLE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(rmsleArr)

  # Output a table with calibration data.
  calibration_data = pandas.DataFrame()
  calibration_data["predictions"] = pandas.Series(predictions)
  calibration_data["targets"] = pandas.Series(targets)
  display.display(calibration_data.describe())

  print("Final RMSLE (on training data): %0.2f" % rmsle)
  plt.show()


train_model(
    learning_rate=0.1,
    steps=500,
    batch_size=5
)
