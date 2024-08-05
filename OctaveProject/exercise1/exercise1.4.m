% Given dataV (including NaN values)
dataV = [
    -0.3999, NaN, -1.0106;
    0.6900, 0.2573, 0.6145;
    0.8156, -1.0565, 0.5077;
    0.7119, NaN, NaN;
    NaN, -0.8051, 0.5913;
    0.6686, 0.5287, -0.6436;
    1.1908, 0.2193, 0.3803;
    NaN, -0.9219, -1.0091;
    -0.0198, NaN, -0.0195;
    -0.1567, -0.0592, -0.0482
];

% Function to compute mean of each column ignoring NaNs
function mean_values = nanmean_columnwise(matrix)
    [nrows, ncols] = size(matrix);
    mean_values = zeros(1, ncols);
    for col = 1:ncols
        valid_elements = matrix(~isnan(matrix(:, col)), col);
        if isempty(valid_elements)
            mean_values(col) = NaN;
        else
            mean_values(col) = mean(valid_elements);
        end
    end
end

% Calculate mean along columns (ignoring NaN values)
mean_values = nanmean_columnwise(dataV);
disp("Mean values along columns (excluding NaN):");
disp(mean_values);

% Remove rows containing NaN values
cleaned_data_rows = dataV(~any(isnan(dataV), 2), :);

% Plotting the cleaned data
figure;
plot(cleaned_data_rows);
title('Removing Rows with NaNs');
xlabel('Index');
ylabel('Value');
grid on;

% Remove columns containing NaN values
cleaned_data_cols = dataV(:, ~any(isnan(dataV), 1));

% Plotting the data after removing columns with NaNs
figure;
plot(cleaned_data_cols);
title('Removing Columns with NaNs');
xlabel('Index');
ylabel('Value');
grid on;

% Replace NaN values with 0
dataV_filled_0 = dataV;
dataV_filled_0(isnan(dataV_filled_0)) = 0;

% Plotting the data after replacing NaNs with 0
figure;
plot(dataV_filled_0);
title('Replacing NaNs with 0');
xlabel('Index');
ylabel('Value');
grid on;

% Find NaN values in dataV
notNaN = ~isnan(dataV);

% Calculate the total number of non-NaN values in each column
totalNo = sum(notNaN, 1);

% Calculate the sum of values in each column
columnTot = sum(dataV, 1);

% Calculate the mean value of each column (excluding NaN values)
colMean = columnTot ./ totalNo;

% Replace NaN values with the column mean values
for i = 1:size(dataV, 2)
    dataV(isnan(dataV(:, i)), i) = colMean(i);
end

% Plotting the data after replacing NaNs with column means
figure;
plot(dataV);
title('Replacing NaNs with Column Means');
xlabel('Index');
ylabel('Value');
grid on;

