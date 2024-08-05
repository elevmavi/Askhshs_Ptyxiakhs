pkg load statistics;

% Define a function for one-hot encoding
function one_hot_labels = one_hot_encode(labels, num_classes)
  num_labels = length(labels);
  one_hot_labels = zeros(num_classes, num_labels);
  for i = 1:num_labels
    one_hot_labels(labels(i) + 1, i) = 1; % Add 1 for 1-based indexing in Octave
  end
end

function output = conv2d(input, kernel)
  [input_rows, input_cols, input_channels, num_images] = size(input);
  [kernel_rows, kernel_cols, kernel_channels, num_kernels] = size(kernel);

  % Output dimensions
  output_rows = input_rows - kernel_rows + 1;
  output_cols = input_cols - kernel_cols + 1;

  % Initialize output
  output = zeros(output_rows, output_cols, num_kernels, num_images);

  for i = 1:num_images
    for j = 1:num_kernels
      for k = 1:input_channels
        % Perform 2D convolution
        output(:, :, j, i) = output(:, :, j, i) + conv2(input(:, :, k, i), kernel(:, :, k, j), 'valid');
      end
    end
  end
end

function output = relu(x)
  output = max(0, x);
end

function output = batch_norm(x, gamma, beta, epsilon)
  mu = mean(x, [1, 2, 3]);
  sigma2 = var(x, 0, [1, 2, 3]);
  x_hat = (x - mu) ./ sqrt(sigma2 + epsilon);
  output = gamma .* x_hat + beta;
end

function output = max_pooling(input, pool_size, stride)
  [input_rows, input_cols, num_channels, num_images] = size(input);
  [pool_rows, pool_cols] = deal(pool_size(1), pool_size(2));

  % Calculate output dimensions
  output_rows = floor((input_rows - pool_rows) / stride) + 1;
  output_cols = floor((input_cols - pool_cols) / stride) + 1;

  output = zeros(output_rows, output_cols, num_channels, num_images);

  for i = 1:num_images
    for j = 1:num_channels
      for r = 1:output_rows
        for c = 1:output_cols
          row_start = (r - 1) * stride + 1;
          col_start = (c - 1) * stride + 1;
          row_end = row_start + pool_rows - 1;
          col_end = col_start + pool_cols - 1;

          % Extract the current pool window
          window = input(row_start:row_end, col_start:col_end, j, i);

          % Apply max pooling using `max` function
           output(r, c, j, i) = max(window(:));
        end
      end
    end
  end
end

function output = flatten(input)
  [rows, cols, channels, num_images] = size(input);
  output = reshape(input, [rows * cols * channels, num_images]);
end

function output = dense(input, weights, biases)
  output = weights * input + biases;
end

function output = softmax(x)
  exps = exp(x - max(x, [], 1));
  output = exps ./ sum(exps, 1);
end

function loss = categorical_crossentropy(predictions, labels)
  % Assume predictions is [num_classes, num_samples]
  % Assume labels is [num_classes, num_samples] (one-hot encoded)
  epsilon = 1e-15; % Small constant to avoid log(0)

  % Clip predictions to avoid log(0)
  predictions = max(min(predictions, 1 - epsilon), epsilon);

  % Compute categorical cross-entropy
  loss = -sum(sum(labels .* log(predictions))) / size(labels, 2);
end

function [weights, biases] = sgd_update(weights, biases, gradients, learning_rate)
  % Update weights and biases using Stochastic Gradient Descent
  % gradients should be a struct with fields for weights and biases
  weights = weights - learning_rate * gradients.weights;
  biases = biases - learning_rate * gradients.biases;
end


% Load the data
d = load('resources/data/mnist.mat');
% Reshape train_images from [60000, 784] to [60000, 28, 28, 1]
train_images = reshape(d.trainX', [28, 28, 1, 60000]);
% Transpose dimensions to [28, 28, 1, 60000] for consistency
train_images = permute(train_images, [2, 1, 3, 4]);

% Normalize train_images
train_images = double(train_images) / 255;

% Reshape test_images from [10000, 784] to [10000, 28, 28, 1]
test_images = reshape(d.testX', [28, 28, 1, 10000]);
% Transpose dimensions to [28, 28, 1, 10000] for consistency
test_images = permute(test_images, [2, 1, 3, 4]);

% Normalize test_images
test_images = double(test_images) / 255;

% Number of classes in MNIST
num_classes = 10;

% One-hot encode the labels
train_labels = one_hot_encode(d.trainY, num_classes);
test_labels = one_hot_encode(d.testY, num_classes);

% Define the size of the validation set
val_size = round(0.25 * size(train_images, 4));

% Create a random permutation of indices
indices = randperm(size(train_images, 4));

% Split the indices for training and validation
val_indices = indices(1:val_size);
train_indices = indices(val_size+1:end);

% Extract validation and training data
val_images = train_images(:, :, :, 1000);
val_labels = train_labels(:, 1000);
train_images = train_images(:, :, :, train_indices);
train_labels = train_labels(:, train_indices);

% Example dimensions and random initialization
input_size = [28, 28, 1, 10]; % Input size [height, width, channels, num_samples]
conv1_filters = 8;
conv2_filters = 16;
conv3_filters = 32;

% Randomly initialize input data and kernel weights
input = rand(input_size);
kernel1 = rand(3, 3, 1, conv1_filters);
kernel2 = rand(3, 3, conv1_filters, conv2_filters);
kernel3 = rand(3, 3, conv2_filters, conv3_filters);

% Randomly initialize BatchNorm parameters
gamma1 = ones(1, 1, conv1_filters);
beta1 = zeros(1, 1, conv1_filters);
gamma2 = ones(1, 1, conv2_filters);
beta2 = zeros(1, 1, conv2_filters);
gamma3 = ones(1, 1, conv3_filters);
beta3 = zeros(1, 1, conv3_filters);
epsilon = 1e-5;

% Convolution + BatchNorm + ReLU + MaxPooling (Layer 1)
conv1_output = conv2d(input, kernel1);
bn1_output = batch_norm(conv1_output, gamma1, beta1, epsilon);
relu1_output = relu(bn1_output);
pool1_output = max_pooling(relu1_output, [2, 2], 2);

% Convolution + BatchNorm + ReLU + MaxPooling (Layer 2)
conv2_output = conv2d(pool1_output, kernel2);
bn2_output = batch_norm(conv2_output, gamma2, beta2, epsilon);
relu2_output = relu(bn2_output);
pool2_output = max_pooling(relu2_output, [2, 2], 2);

% Convolution + BatchNorm + ReLU (Layer 3)
conv3_output = conv2d(pool2_output, kernel3);
bn3_output = batch_norm(conv3_output, gamma3, beta3, epsilon);
relu3_output = relu(bn3_output);

% Flatten
flattened_output = flatten(relu3_output);

% Initialize Dense Layer Weights and Biases
dense_weights = rand(num_classes, numel(flattened_output) / size(input, 4));
dense_biases = rand(num_classes, 1);

% Dense Layer
dense_output = dense(flattened_output, dense_weights, dense_biases);

% Softmax Activation, contains the probabilities for each class
final_output = softmax(dense_output);

% Define hyperparameters
learning_rate = 0.01;
num_epochs = 1;
batch_size = 1; % Adjust based on available memory

% Number of training samples
num_train_samples = size(train_images, 1);

% Dense Layer Initialization
num_flattened = 288; % Example size; adjust based on actual output size
dense_weights = rand(num_classes, num_flattened);
dense_biases = rand(num_classes, 1);

for epoch = 1:num_epochs
  % Shuffle training data
  indices = randperm(num_train_samples);
  train_images = train_images(:, :, :, indices);
  train_labels = train_labels(:, indices);

  % Iterate over batches
  for start_idx = 1:batch_size:num_train_samples
    end_idx = min(start_idx + batch_size - 1, num_train_samples);

    % Get batch data
    batch_images = train_images(:, :, :, start_idx:end_idx);
    batch_labels = train_labels(:, start_idx:end_idx);

    % Forward Pass
    conv1_output = conv2d(batch_images, kernel1);
    bn1_output = batch_norm(conv1_output, gamma1, beta1, epsilon);
    relu1_output = relu(bn1_output);
    pool1_output = max_pooling(relu1_output, [2, 2], 2);

    conv2_output = conv2d(pool1_output, kernel2);
    bn2_output = batch_norm(conv2_output, gamma2, beta2, epsilon);
    relu2_output = relu(bn2_output);
    pool2_output = max_pooling(relu2_output, [2, 2], 2);

    conv3_output = conv2d(pool2_output, kernel3);
    bn3_output = batch_norm(conv3_output, gamma3, beta3, epsilon);
    relu3_output = relu(bn3_output);

    flattened_output = flatten(relu3_output);

    dense_output = dense(flattened_output, dense_weights, dense_biases);
    final_output = softmax(dense_output);

    % Compute Loss
    loss = categorical_crossentropy(final_output, batch_labels);
    fprintf('Epoch %d, Batch %d, Loss: %f\n', epoch, start_idx, loss);

    % Backward Pass (Simplified, no actual gradient computation)
    % You need to implement the actual gradient calculations for weights and biases
    gradients.weights = rand(size(dense_weights));
    gradients.biases = rand(size(dense_biases));

    % Update Parameters
    [dense_weights, dense_biases] = sgd_update(dense_weights, dense_biases, gradients, learning_rate);
  end

  % Validation
  val_conv1_output = conv2d(val_images, kernel1);
  val_bn1_output = batch_norm(val_conv1_output, gamma1, beta1, epsilon);
  val_relu1_output = relu(val_bn1_output);
  val_pool1_output = max_pooling(val_relu1_output, [2, 2], 2);

  val_conv2_output = conv2d(val_pool1_output, kernel2);
  val_bn2_output = batch_norm(val_conv2_output, gamma2, beta2, epsilon);
  val_relu2_output = relu(val_bn2_output);
  val_pool2_output = max_pooling(val_relu2_output, [2, 2], 2);

  val_conv3_output = conv2d(val_pool2_output, kernel3);
  val_bn3_output = batch_norm(val_conv3_output, gamma3, beta3, epsilon);
  val_relu3_output = relu(val_bn3_output);

  val_flattened_output = flatten(val_relu3_output);

  val_dense_output = dense(val_flattened_output, dense_weights, dense_biases);
  val_final_output = softmax(val_dense_output);

  val_loss = categorical_crossentropy(val_final_output, val_labels);
  fprintf('Epoch %d, Validation Loss: %f\n', epoch, val_loss);

  % Save training and validation loss
  history.train_loss(epoch) = loss;
  history.val_loss(epoch) = val_loss;
end

test_images = test_images(:, :, :, 1:200); %reduce data for faster execution
% Evaluate the model on the test data
test_conv1_output = conv2d(test_images, kernel1);
test_bn1_output = batch_norm(test_conv1_output, gamma1, beta1, epsilon);
test_relu1_output = relu(test_bn1_output);
test_pool1_output = max_pooling(test_relu1_output, [2, 2], 2);

test_conv2_output = conv2d(test_pool1_output, kernel2);
test_bn2_output = batch_norm(test_conv2_output, gamma2, beta2, epsilon);
test_relu2_output = relu(test_bn2_output);
test_pool2_output = max_pooling(test_relu2_output, [2, 2], 2);

test_conv3_output = conv2d(test_pool2_output, kernel3);
test_bn3_output = batch_norm(test_conv3_output, gamma3, beta3, epsilon);
test_relu3_output = relu(test_bn3_output);

test_flattened_output = flatten(test_relu3_output);

test_dense_output = dense(test_flattened_output, dense_weights, dense_biases);
test_final_output = softmax(test_dense_output);

test_labels = test_labels(:, 1:200); %reduce data for faster execution
test_loss = categorical_crossentropy(test_final_output, test_labels);

[~, test_predicted_labels] = max(test_final_output, [], 1);
[~, test_true_labels] = max(test_labels, [], 1);
test_accuracy = mean(test_predicted_labels == test_true_labels);

fprintf('Test accuracy: %.2f%%\n', test_accuracy * 100);

% Predict labels for the validation set
val_predictions = val_final_output;
[~, val_predicted_labels] = max(val_predictions, [], 1);
[~, val_true_labels] = max(val_labels, [], 1);

% Calculate accuracy on validation set
val_accuracy = mean(val_predicted_labels == val_true_labels);
fprintf('Validation accuracy: %.2f%%\n', val_accuracy * 100);


