pkg load statistics;  % Load the statistics package

% Define the data matrix A and labels B
A = [
    1, 0, 0, 0, 0, 0, 1, 1;
    1, 0, 0, 0, 0, 1, 0, 0;
    0, 1, 0, 1, 1, 1, 1, 0;
    0, 0, 0, 1, 1, 1, 0, 1;
    0, 0, 1, 1, 1, 0, 0, 0;
    1, 0, 0, 0, 0, 0, 1, 0
];
B = {'recipe', 'physics', 'travel', 'hotel', 'travel', 'recipe'};

% Compute pairwise cosine distances
function d = cosine_distance(X)
  normX = sqrt(sum(X .^ 2, 2));
  d = pdist(X, 'cosine');
end

cosine_distances = cosine_distance(A);

% Perform hierarchical clustering using single linkage
Z_single_cosine = linkage(cosine_distances, 'single');

% Plot dendrogram
figure;
dendrogram(Z_single_cosine, 'labels', B);
title('Hierarchical Single Linkage Clustering Dendrogram (Cosine Distance)');
xlabel('Documents');
ylabel('Distance');
set(gca, 'XTickLabelRotation', 45);  % Rotate x-axis labels

% Compute pairwise Jaccard distances
function d = jaccard_distance(X)
  d = pdist(X, 'jaccard');
end

jaccard_distances = jaccard_distance(A);

% Perform hierarchical clustering using single linkage
Z_single_jaccard = linkage(jaccard_distances, 'single');

% Plot dendrogram
figure;
dendrogram(Z_single_jaccard, 'labels', B);
title('Hierarchical Single Linkage Clustering Dendrogram (Jaccard Distance)');
xlabel('Documents');
ylabel('Distance');
set(gca, 'XTickLabelRotation', 45);  % Rotate x-axis labels

% Perform hierarchical clustering using average linkage
Z_avg_cosine = linkage(cosine_distances, 'average');

% Plot dendrogram
figure;
dendrogram(Z_avg_cosine, 'labels', B);
title('Hierarchical Average Linkage Clustering Dendrogram (Cosine Distance)');
xlabel('Documents');
ylabel('Distance');
set(gca, 'XTickLabelRotation', 45);  % Rotate x-axis labels

Z_avg_jaccard = linkage(jaccard_distances, 'average');

% Plot dendrogram
figure;
dendrogram(Z_avg_jaccard, 'labels', B);
title('Hierarchical Average Linkage Clustering Dendrogram (Jaccard Distance)');
xlabel('Documents');
ylabel('Distance');
set(gca, 'XTickLabelRotation', 45);  % Rotate x-axis labels

