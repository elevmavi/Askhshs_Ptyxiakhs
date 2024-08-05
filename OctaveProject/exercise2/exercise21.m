pkg load statistics;  % Load the statistics package for distance functions
pkg load io;  % Load the statistics package for distance functions

function [D1, D2, D3, D4, D5, D6, D7] = compute_distances(X)
    % Compute pairwise distances between data points using various distance metrics.
    % X: Input data matrix of size (n_samples, n_features).
    % Returns a tuple of distance matrices corresponding to different distance metrics.

    % Number of samples
    n = size(X, 1);

    % Compute pairwise distances using various methods
    D1 = pdist(X, 'euclidean');  % Compute pairwise Euclidean distances
    V = std(X, 0, 1);  % Compute standard deviation for each feature (column)
    D2 = pdist(X, 'seuclidean', V);  % Compute pairwise standardized Euclidean distances
    D3 = pdist(X, 'cityblock');  % Manhattan (cityblock) distance
    D4 = pdist(X, 'minkowski', 3);  % Minkowski distance with p=3
    D5 = custom_pdist(X, @chebyshev_dist_fun);  % Chebyshev distance
    % Mahalanobis distance
    S = cov(X);  % Covariance matrix
    invS = inv(S);  % Inverse of covariance matrix
    D6 = custom_pdist(X, @(u, v) mahalanobis_distance(u, v, invS));  % Mahalanobis distance
    D7 = pdist(X, 'cosine');  % Cosine distance

end

function D = custom_pdist(X, dist_fun)
    % Custom implementation of pdist
    % X: Input data matrix of size (n_samples, n_features)
    % dist_fun: Function handle for distance computation
    % Returns a condensed distance matrix

    [n, ~] = size(X);
    num_distances = n * (n - 1) / 2;
    D = zeros(num_distances, 1);
    index = 1;
    for i = 1:n-1
        for j = i+1:n
            D(index) = dist_fun(X(i, :), X(j, :));
            index = index + 1;
        end
    end
end

function d = chebyshev_dist_fun(u, v)
    d = max(abs(u - v));
end

% Helper function for Mahalanobis distance
function d = mahalanobis_distance(u, v, invS)
    d = sqrt((u - v) * invS * (u - v)');  % Mahalanobis distance formula
end

function compute_and_plot_distances(X, i, j)
    % Compute and visualize pairwise distances between specified observations.
    % X: Input data matrix of size (n_samples, n_features).
    % i: Index of the first observation.
    % j: Index of the second observation.

    % Compute distances
    [D1, D2, D3, D4, D5, D6, D7] = compute_distances(X);

    % Convert condensed distance matrices to square distance matrices
    D1_sq = squareform(D1);
    D2_sq = squareform(D2);
    D3_sq = squareform(D3);
    D4_sq = squareform(D4);
    D5_sq = squareform(D5);
    D6_sq = squareform(D6);
    D7_sq = squareform(D7);

    % Select distances for observations i and j
    distances = [D1_sq(i, j), D2_sq(i, j), D3_sq(i, j), D4_sq(i, j), D5_sq(i, j), D6_sq(i, j), D7_sq(i, j)];

    % Find the method yielding the maximum and minimum distances
    methods = {'euclidean', 'seuclidean', 'cityblock', 'minkowski', 'chebyshev', 'mahalanobis', 'cosine'};
    [max_distance, max_index] = max(distances);
    [min_distance, min_index] = min(distances);
    max_distance_method = methods{max_index};
    min_distance_method = methods{min_index};

    % Create the figure
    figure;
    % Plot the bar chart
    bar(distances);
    % Set the x-axis tick labels
    set(gca, 'XTickLabel', methods, 'XTick', 1:length(methods));
    % Rotate the x-axis tick labels for better readability
    xtickangle(45);
    % Set the title and labels
    title(sprintf('Distances between observations %d and %d', i, j));
    xlabel('Distance Metric');
    ylabel('Distance Value');
    % Show the plot
    grid on;

    % Print maximum and minimum distances and their corresponding methods
    printf('Maximum distance between observations %d and %d (%.2f) using method: %s\n', i, j, max_distance, max_distance_method);
    printf('Minimum distance between observations %d and %d (%.2f) using method: %s\n', i, j, min_distance, min_distance_method);
end

% Create a random matrix with 100 rows and 5 columns
X = randn(100, 5);

% Compute and plot distances for observations 24 and 75
i = 24;
j = 75;
compute_and_plot_distances(X, i, j);

% Compute and plot distances for observations 1 and 100
i = 1;
j = 100;
compute_and_plot_distances(X, i, j);

