% Load the Iris dataset
data = load('resources/data/iris.txt');

% Extract petal length and petal width
X = data(:, [3, 4]);

% Parameters for DBSCAN
epsilon = 0.2;
min_samples = 4;

% DBSCAN algorithm implementation
function IDX = dbscan(X, epsilon, MinPts)
    C = 0;
    n = size(X, 1);
    IDX = -1 * ones(n, 1); % Initialize cluster indices to -1 (unclassified)
    visited = false(n, 1); % Array to track visited points

    for i = 1:n
        if ~visited(i)
            visited(i) = true;
            Neighbors = [];
            for j = 1:n
                if norm(X(i, :) - X(j, :)) <= epsilon
                    Neighbors = [Neighbors; j];
                end
            end
            if numel(Neighbors) < MinPts
                IDX(i) = 0; % Mark as noise
            else
                C += 1;
                IDX(i) = C;
                k = 1;
                while true
                    j = Neighbors(k);
                    if ~visited(j)
                        visited(j) = true;
                        Neighbors2 = [];
                        for i = 1:n
                            if norm(X(j, :) - X(i, :)) <= epsilon
                                Neighbors2 = [Neighbors2; i];
                            end
                        end
                    end
                    if IDX(j) == -1
                        IDX(j) = C; % Assign cluster ID
                    end
                    k += 1;
                    if k > numel(Neighbors)
                        break;
                    end
                end
            end
        end
    end
end

% Plot cluster results
function plot_cluster_result(X, labels, epsilon, min_samples)
    unique_labels = unique(labels);

    figure;
    hold on;
    for i = 1:length(unique_labels)
        label = unique_labels(i);
        if label == -1
            % Plot noise points as black
            scatter(X(labels == label, 1), X(labels == label, 2), 'k', 'filled', 'DisplayName', 'Noise');
        else
            % Plot points for each cluster
            scatter(X(labels == label, 1), X(labels == label, 2), 'filled', 'DisplayName', sprintf('Cluster %d', label));
        end
    end
    title(sprintf('DBSCAN Clustering (ε = %.2f, MinPts = %d)', epsilon, min_samples));
    xlabel('Petal Length');
    ylabel('Petal Width');
    legend;
    hold off;
end

% Perform DBSCAN clustering
clusters = dbscan(X, epsilon, min_samples);

% Scatter plot of the clusters (Figure 1)
figure;
scatter(X(:, 1), X(:, 2), 50, clusters, 'filled');
title('DBSCAN Clustering');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
colorbar;
colormap(jet);

% Set color axis limits for clusters (handle noise case)
cluster_min = min(clusters);
cluster_max = max(clusters);
if cluster_min == cluster_max
    caxis([cluster_min - 1, cluster_min + 1]); % Set a small range if all values are the same
else
    caxis([cluster_min, cluster_max]);
end

% Plot cluster results using custom function (Figure 2)
plot_cluster_result(X, clusters, epsilon, min_samples);

% Z-score normalization
X_zscore = (X - mean(X)) ./ std(X);

% Perform DBSCAN clustering on normalized data
clusters_zscore = dbscan(X_zscore, epsilon, min_samples);

% Plot the clusters with normalized data (Figure 3)
figure;
scatter(X_zscore(:, 1), X_zscore(:, 2), 50, clusters_zscore, 'filled');
title('DBSCAN Clustering (Z-score Normalized)');
xlabel('Petal Length (Z-score)');
ylabel('Petal Width (Z-score)');
colorbar;
colormap(jet);

% Set color axis limits for Z-score normalized clusters (handle noise case)
zscore_min = min(clusters_zscore);
zscore_max = max(clusters_zscore);
if zscore_min == zscore_max
    caxis([zscore_min - 1, zscore_min + 1]); % Set a small range if all values are the same
else
    caxis([zscore_min, zscore_max]);
end

