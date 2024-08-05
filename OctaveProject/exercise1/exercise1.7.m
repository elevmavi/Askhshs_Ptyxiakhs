% Load the dataset (assuming it's a CSV file)
try
    iris = csvread('resources/data/iris.txt');
catch
    error('Failed to load the dataset. Check the file path and format.');
end

% Create a copy of the original iris dataset
irisV = iris;

% Introduce NaN values into the iris dataset
[ro, co] = size(iris);
p1 = 60;  % Percentage of NaN values to introduce
p = round(p1 * ro / 100);

% Create random indices to introduce NaN values
r1 = randperm(ro);

% Replace random rows with NaN values in each column
for i = 1:4
    irisV(r1(1:p), i) = NaN;
end

% Remove rows containing NaN values
data = irisV(~any(isnan(irisV), 2), :);

% Plotting the cleaned data
figure;
plot(data(:, 1:end));
title('Removing Rows with NaNs');
xlabel('Index');
ylabel('Value');
grid on;

% Remove columns containing NaN values
data = irisV(:, ~any(isnan(irisV), 1));

% Plotting the data after removing columns with NaNs
figure;
plot(data(:, 1:end));
title('Removing Columns with NaNs');
xlabel('Index');
ylabel('Value');
grid on;

% Replace NaN values with 0
data = irisV;
data(isnan(data)) = 0;

% Plotting the data after replacing NaNs with 0
figure;
plot(data(:, 1:end));
title('Replacing NaNs with 0');
xlabel('Index');
ylabel('Value');
grid on;

% Find NaN values in irisV
notNaN = ~isnan(irisV);

% Replace NaN values with 0
irisV(isnan(irisV)) = 0;

% Calculate the total number of non-NaN values in each column
totalNo = sum(notNaN);

% Calculate the sum of values in each column
columnTot = sum(irisV);

% Calculate the mean value of each column (excluding NaN values)
colMean = columnTot ./ totalNo;

% Replace NaN values with the column mean values
for i = 1:size(irisV, 2)
    irisV(isnan(irisV(:, i)), i) = colMean(i);
end

% Plotting the data after replacing NaNs with column means
figure;
plot(irisV(:, 1:4));
title('Replacing NaNs with Column Means');
xlabel('Index');
ylabel('Value');
grid on;

