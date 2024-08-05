% Load iris dataset
iris = dlmread('resources/data/iris.txt', ',');

% Define iris species based on the fifth column (species code)
setosa = iris(iris(:, 5) == 1, 1:4);  % data for setosa
versicolor = iris(iris(:, 5) == 2, 1:4);  % data for versicolor
virginica = iris(iris(:, 5) == 3, 1:4);  % data for virginica

% Characteristics of iris (features)
characteristics = {'sepal length', 'sepal width', 'petal length', 'petal width'};

% Pairs of characteristics for scatter plots
pairs = [1, 2; 1, 3; 1, 4; 2, 3; 2, 4; 3, 4];

% Create a figure with subplots
figure;

for i = 1:size(pairs, 1)
    x = pairs(i, 1);
    y = pairs(i, 2);

    % Create subplot
    subplot(2, 3, i);

    % Plot data
    plot(setosa(:, x), setosa(:, y), 'r.', 'DisplayName', 'setosa'); hold on;
    plot(versicolor(:, x), versicolor(:, y), 'g.', 'DisplayName', 'versicolor');
    plot(virginica(:, x), virginica(:, y), 'b.', 'DisplayName', 'virginica');

    % Set labels and title
    xlabel(characteristics{x});
    ylabel(characteristics{y});
    legend('show');
    hold off;
end


