pkg load io;  % Load the io package for file operations

function normalized_data = linear_transform(data)
  % Compute min and max values for each column
  min_vals = min(data);
  max_vals = max(data);

  % Linear transformation (min-max scaling) for each column
  normalized_data = (data - min_vals) ./ (max_vals - min_vals);
endfunction

function normalized_data = zscore_transform(data)
  % Compute mean and standard deviation for each column
  mean_vals = mean(data);
  std_vals = std(data);

  % Z-score transformation for each column
  normalized_data = (data - mean_vals) ./ std_vals;
endfunction

% Load iris dataset
iris = csvread('resources/data/iris.txt');
iris = iris(:, 1:4);  % Assuming the first four columns are needed
irisV = iris;  % Create a copy of the original iris dataset

% Perform linear normalization
iris_linear_normalized = linear_transform(iris);

% Plotting the linearly normalized data
figure;
plot(iris_linear_normalized);
title('Linear Normalization (Min-Max Scaling)');
xlabel('Index');
ylabel('Normalized Value');
grid on;

% Perform z-score normalization
iris_zscore_normalized = zscore_transform(irisV);

% Plotting the z-score normalized data
figure;
plot(iris_zscore_normalized);
title('Z-score Normalization');
xlabel('Index');
ylabel('Normalized Value');
grid on;

% Given data (from MATLAB)
data = [
  -0.3999, -0.2625, -1.0106;
   0.6900,  0.2573,  0.6145;
   0.8156, -1.0565,  0.5077;
   0.7119, -0.2625, -0.0708;
   0.4376, -0.8051,  0.5913;
   0.6686,  0.5287, -0.6436;
   1.1908,  0.2193,  0.3803;
   0.4376, -0.9219, -1.0091;
  -0.0198, -0.2625, -0.0195;
  -0.1567, -0.0592, -0.0482
];
dataV = data;  % Create a copy of the original data

% Perform linear normalization
data_linear_normalized = linear_transform(data);

% Plotting the linearly normalized data
figure;
plot(data_linear_normalized);
title('Linear Normalization (Min-Max Scaling)');
xlabel('Index');
ylabel('Normalized Value');
grid on;

% Perform z-score normalization
data_zscore_normalized = zscore_transform(dataV);

% Plotting the z-score normalized data
figure;
plot(data_zscore_normalized);
title('Z-score Normalization');
xlabel('Index');
ylabel('Normalized Value');
grid on;

