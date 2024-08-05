% Load the data
data = load('resources/data/xV.mat');
xV = data.xV;

% Ensure xV is a matrix and check its dimensions
if !ismatrix(xV)
    error('xV should be a matrix.');
end

% Sanitize xV by replacing NaN and Inf values with 0
xV(isnan(xV) | isinf(xV)) = 0;

% Extract features after sanitization
feature1 = xV(:, [1, 2]);
feature2 = xV;
feature3 = xV(:, [297, 306]);  % Octave is 1-based indexing

% Define the number of clusters
k = 3;

% Check dimensions and validity
[numRowsFeature1, ~] = size(feature1);
[numRowsFeature2, ~] = size(feature2);
[numRowsFeature3, ~] = size(feature3);

if k <= 0 || k > numRowsFeature1
    error('Number of clusters k must be greater than 0 and less than or equal to the number of data points in feature1.');
end
if k <= 0 || k > numRowsFeature2
    error('Number of clusters k must be greater than 0 and less than or equal to the number of data points in feature2.');
end
if k <= 0 || k > numRowsFeature3
    error('Number of clusters k must be greater than 0 and less than or equal to the number of data points in feature3.');
end

% Apply k-means algorithm
try
    [IDX1, C1] = kmeans(feature1, k);
    [IDX2, C2] = kmeans(feature2, k);
    [IDX3, C3] = kmeans(feature3, k);
catch err
    error('An error occurred during k-means clustering: %s', err.message);
end

% Plot for feature1
figure;
hold on;
for cluster = 1:k
    scatter(feature1(IDX1 == cluster, 1), feature1(IDX1 == cluster, 2), 12, 'DisplayName', sprintf('Cluster %d', cluster));
end
scatter(C1(:, 1), C1(:, 2), 200, 'kx', 'LineWidth', 2, 'DisplayName', 'Centroids');
title('Clustering Result with Centroids for Feature 1');
xlabel('Feature 1');
ylabel('Feature 2');
legend;
hold off;

% Plot for feature2
figure;
hold on;
for cluster = 1:k
    scatter(feature2(IDX2 == cluster, 1), feature2(IDX2 == cluster, 2), 12, 'DisplayName', sprintf('Cluster %d', cluster));
end
scatter(C2(:, 1), C2(:, 2), 200, 'kx', 'LineWidth', 2, 'DisplayName', 'Centroids');
title('Clustering Result with Centroids for Feature 2');
xlabel('Feature 1');
ylabel('Feature 2');
legend;
hold off;

% Plot for feature3
figure;
hold on;
for cluster = 1:k
    scatter(feature3(IDX3 == cluster, 1), feature3(IDX3 == cluster, 2), 12, 'DisplayName', sprintf('Cluster %d', cluster));
end
scatter(C3(:, 1), C3(:, 2), 200, 'kx', 'LineWidth', 2, 'DisplayName', 'Centroids');
title('Clustering Result with Centroids for Feature 3');
xlabel('Feature 1');
ylabel('Feature 2');
legend;
hold off;

