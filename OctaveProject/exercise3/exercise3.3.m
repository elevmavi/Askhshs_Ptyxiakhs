pkg load statistics; % Load the statistics package for clustering functions

% Load the data from .mat file
data = load('resources/data/xV.mat');
xV = data.xV;

% Reshape to (600, 1)
first_feature = xV(:, 1);
% Calculate pairwise Euclidean distances for the first feature
distances_first_feature = pdist(first_feature);

% Perform hierarchical clustering with single linkage
Z_single_euclidean = linkage(distances_first_feature, 'single');

% Extract cluster labels for different numbers of clusters (2 to 10)
for n_clusters = 2:10
    % Use cluster to obtain cluster labels
    cluster_labels = cluster(Z_single_euclidean, 'maxclust', n_clusters);

    % Convert cluster labels to a space-separated string
    labels_str = sprintf('%d ', cluster_labels);

    % Print cluster labels for the current number of clusters
    printf("Cluster labels for 1st column Single Linkage %d clusters (Euclidean Distance): %s\n", n_clusters, labels_str);
end

second_feature = xV(:, :);

% Standardize xV
xV_standardized = zscore(second_feature);
% Replace NaN values with 0 in standardized data
nan_mask = isnan(xV_standardized);
xV_standardized(nan_mask) = 0;

% Calculate pairwise Euclidean distances for all features
distances_xV = pdist(xV_standardized);

% Perform hierarchical clustering with average linkage
Z = linkage(distances_xV, 'average');

% Extract cluster labels for different numbers of clusters (2 to 10)
for n_clusters = 2:10
    % Use cluster to obtain cluster labels
    cluster_labels = cluster(Z, 'maxclust', n_clusters);

    % Convert cluster labels to a space-separated string
    labels_str2 = sprintf('%d ', cluster_labels);

    % Print cluster labels for the current number of clusters
    printf("Cluster labels for all data, with Average Linkage %d clusters (Euclidean Distance): %s\n", n_clusters, labels_str2);
end

