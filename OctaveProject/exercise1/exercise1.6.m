% Load iris dataset
iris = load('resources/data/iris.txt', '-ascii');

% Create a copy of the original iris dataset
irisV = iris;

% Introduce NaN values into the iris dataset
[ro, co] = size(iris);
p1 = 60;  % Percentage of NaN values to introduce
p = floor(p1 * ro / 100);

% Create random indices to introduce NaN values
r1 = randperm(ro);

% Replace random rows with NaN values in each column
for i = 1:4
    irisV(r1(1:p), i) = NaN;
end

% Plot the irisV dataset (scatter plot excluding NaN values)
figure;
% Remove rows with NaN values in the first two columns for plotting
valid_rows = ~isnan(irisV(:, 1)) & ~isnan(irisV(:, 2));
scatter(irisV(valid_rows, 1), irisV(valid_rows, 2), [], irisV(valid_rows, 5), 'filled');
xlabel('Sepal Length');
ylabel('Sepal Width');
title('Scatter Plot of Iris Dataset (with NaN Values)');

% Add colorbar and set the color limits manually
c = colorbar;
% Adjust color limits to match the data range
caxis([min(iris(:,5)), max(iris(:,5))]);

% Add a label manually to the colorbar
ylabel(c, 'Species');

grid on;

