import pandas
import numpy
import tensorflow
import math
from tensorflow.python.data import Dataset
from sklearn import metrics

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

dataframe = pandas.read_csv("train.csv")
dataframe = dataframe.reindex(numpy.random.permutation(dataframe.index))

features = dataframe[["OverallQual"]]
feature_columns = [tensorflow.feature_column.numeric_column("OverallQual")]

targets = dataframe["SalePrice"]

my_optimizer=tensorflow.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tensorflow.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tensorflow.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # Convert pandas data into a dict of np arrays.
    features = {key:numpy.array(value) for key,value in dict(features).items()}

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

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = numpy.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)

min_SalePrice = dataframe["SalePrice"].min()
max_SalePrice = dataframe["SalePrice"].max()
min_max_difference = max_SalePrice - min_SalePrice

print("Min. SalePrice Value: %0.3f" % min_SalePrice)
print("Max. SalePrice Value: %0.3f" % max_SalePrice)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)


def train_model(learning_rate, steps, batch_size, input_feature="OverallQual"):
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
  colors = [cm.coolwarm(x) for x in numpy.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = numpy.array([item['predictions'][0] for item in predictions])

    # Compute loss.
    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = numpy.array([0, sample[my_label].max()])

    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = numpy.maximum(numpy.minimum(x_extents,
                                      sample[features].max()),
                           sample[features].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period])
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)

  # Output a table with calibration data.
  calibration_data = pandas.DataFrame()
  calibration_data["predictions"] = pandas.Series(predictions)
  calibration_data["targets"] = pandas.Series(targets)
  display.display(calibration_data.describe())

  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
  plt.show()


train_model(
    learning_rate=40,
    steps=500,
    batch_size=5
)

"""
sample = dataframe.sample(n=300)


x_0 = sample["OverallQual"].min()
x_1 = sample["OverallQual"].max()

# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/OverallQual/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted SalePrices for the min and max OverallQual values.
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
plt.ylabel("SalePrice")
plt.xlabel("OverallQual")

# Plot a scatter plot from our data sample.
plt.scatter(sample["OverallQual"], sample["SalePrice"])

# Display graph.
plt.show()
"""
