pkg load statistics;  % Load the statistics package for K-means

% Load data from the text file
iris_data = dlmread('resources/data/iris.txt', ',');

% Extract features
feature1 = iris_data(:, [3, 4]);  % Petal length and petal width (last two columns)
feature2 = iris_data(:, 1:4);     % All four features

% Number of clusters
k = 3;

% Apply K-means algorithm to feature1
[idx1, C1] = kmeans(feature1, k, 'Replicates', 10);

% Plotting the data and clusters for feature1
figure;
hold on;
colors = ['r', 'g', 'b'];
for cluster = 1:k
    scatter(feature1(idx1 == cluster, 1), feature1(idx1 == cluster, 2), 36, colors(cluster), 'filled', 'DisplayName', sprintf('Cluster %d', cluster));
end
scatter(C1(:, 1), C1(:, 2), 100, 'k', 'x', 'DisplayName', 'Centroids');
title('Clustering Result with Centroids (Feature 1)');
xlabel('Feature 1');
ylabel('Feature 2');
legend;
hold off;

% Apply K-means algorithm to feature2
[idx2, C2] = kmeans(feature2, k, 'Replicates', 10);

% Plotting the data and clusters for feature2
figure;
hold on;
for cluster = 1:k
    scatter(feature1(idx2 == cluster, 1), feature1(idx2 == cluster, 2), 36, colors(cluster), 'filled', 'DisplayName', sprintf('Cluster %d', cluster));
end
scatter(C2(:, 1), C2(:, 2), 100, 'k', 'x', 'DisplayName', 'Centroids');
title('Clustering Result with Centroids (Feature 2)');
xlabel('Feature 1');
ylabel('Feature 2');
legend;
hold off;

