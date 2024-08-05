pkg load statistics;

function tabulate_clusters(clusters, species)
    unique_clusters = unique(clusters);
    unique_species = unique(species);

    printf('Clusters vs Species:\n');
    printf('\t');
    for s = 1:length(unique_species)
        printf('%s\t', unique_species{s});
    end
    printf('\n');

    for c = 1:length(unique_clusters)
        printf('Cluster %d\t', unique_clusters(c));
        for s = 1:length(unique_species)
            count = sum(clusters == unique_clusters(c) & strcmp(species, unique_species{s}));
            printf('%d\t', count);
        end
        printf('\n');
    end
end

% Load iris data
iris_data = dlmread('resources/data/iris.txt', ',');

% Extract features and target labels from iris_data
data = iris_data(:, 1:4);  % Features (sepal length, sepal width, petal length, petal width)
species = iris_data(:, 5); % Target labels (species)

% Convert numeric species data to cell array of strings for tabulation
species = arrayfun(@(x) sprintf('Species %d', x), species, 'UniformOutput', false);

% Calculate pairwise distances using Euclidean distance
euclidean_distances = pdist(data, 'euclidean');

% Perform hierarchical clustering using single linkage
Z_single_euclidean = linkage(euclidean_distances, 'single');

% Plot dendrogram for Simple Linkage
figure;
dendrogram(Z_single_euclidean);
title('Hierarchical Clustering Dendrogram (Single Linkage)');
xlabel('Species');
ylabel('Distance');

k = 3;  % Number of clusters
% Assign observations to clusters for Single Linkage
max_cluster_single = cluster(Z_single_euclidean, 'maxclust', k);

% Crosstab to compare predicted clusters with true species labels for irisData
printf("Clustering Result for Single Linkage:\n");
tabulate_clusters(max_cluster_single, species);

% Perform hierarchical clustering using average linkage
Z_average_euclidean = linkage(euclidean_distances, 'average');

% Plot dendrogram for Average Linkage
figure;
dendrogram(Z_average_euclidean);
title('Hierarchical Clustering Dendrogram (Average Linkage)');
xlabel('Species');
ylabel('Distance');

% Assign observations to clusters for Average Linkage
max_cluster_average = cluster(Z_average_euclidean, 'maxclust', k);

% Crosstab to compare predicted clusters with true species labels for irisData
printf("Clustering Result for Average Linkage:\n");
tabulate_clusters(max_cluster_average, species);

% Perform hierarchical clustering using complete linkage
Z_complete_euclidean = linkage(euclidean_distances, 'complete');

% Plot dendrogram for Complete Linkage
figure;
dendrogram(Z_complete_euclidean);
title('Hierarchical Clustering Dendrogram (Complete Linkage)');
xlabel('Species');
ylabel('Distance');

% Assign observations to clusters for Complete Linkage
clusters_complete = cluster(Z_complete_euclidean, 'maxclust', k);

% Crosstab to compare predicted clusters with true species labels for irisData
printf("Clustering Result for Complete Linkage:\n");
tabulate_clusters(clusters_complete, species);

% Define combinations of features (columns) for clustering
feature_combinations = {
    [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4], [1, 2, 3, 4]
};

% Define distance metrics
distance_metrics = {'euclidean', 'cityblock', 'chebychev'};

% Iterate over all combinations of features and distance metrics
for i = 1:length(feature_combinations)
    features = feature_combinations{i};
    data_subset = data(:, features);

    for j = 1:length(distance_metrics)
        metric = distance_metrics{j};
        % Calculate pairwise distances using current metric
        distances_subset = pdist(data_subset, metric);

        % Perform hierarchical clustering using single linkage and current metric
        Z_subset = linkage(distances_subset, 'single');

        % Plot dendrogram for the current clustering
        figure;
        dendrogram(Z_subset);
        title(sprintf('Hierarchical Clustering Dendrogram (Single Linkage, Features %s, Metric %s)', mat2str(features), metric));
        xlabel('Species');
        ylabel('Distance');

        % Assign observations to clusters using single linkage for the current clustering
        clusters_subset = cluster(Z_subset, 'maxclust', k);

        % Compare predicted clusters with true species labels using tabulate_clusters
        printf("Clustering Result for Single Linkage, Features %s, Metric %s:\n", mat2str(features), metric);
        tabulate_clusters(clusters_subset, species);
    end
end

% Define the feature sets for clustering
features_1 = data(:, [3, 4]);  % Features [3, 4]
features_2 = data(:, [1, 2, 3, 4]);  % Features [1, 2, 3, 4]

% Define the number of clusters to evaluate (from 4 to 10 clusters)
num_clusters_range = 4:10;

% Perform hierarchical clustering with single linkage and Euclidean distance for each feature set
for feature_set = {features_1, features_2}
    features = feature_set{1};
    printf("Clustering results for features: %s\n", mat2str(features));
    for num_clusters = num_clusters_range
        % Calculate pairwise distances using Euclidean distance
        distances = pdist(features, 'euclidean');
        Z = linkage(distances, 'single');

        % Assign observations to clusters
        clusters = cluster(Z, 'maxclust', num_clusters);

        % Compute and print the cluster sizes
        cluster_sizes = arrayfun(@(x) sum(clusters == x), 1:num_clusters);
        printf("Number of clusters: %d, Cluster sizes: %s\n", num_clusters, mat2str(cluster_sizes));

        % Plot the dendrogram for visualization (optional)
        figure;
        dendrogram(Z);
        title(sprintf("Dendrogram for %d Clusters using Features: %s", num_clusters, mat2str(features)));
        xlabel('Species');
        ylabel('Distance');
    end
end

