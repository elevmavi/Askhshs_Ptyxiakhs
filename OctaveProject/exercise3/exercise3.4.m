pkg load statistics

% Load the data
data = load('resources/data/enron100.mat');
data = data.en2;  % Replace 'en2' with the actual variable name if different

% Extract the relevant columns
first_two_rows = data(1:2, 2:3);
first_thousand_rows = data(1:1000, 2:3);
%all_data = data(:, 2:3); should be like this but too slow
all_data = data(1:3000, 2:3);

% Function to compute Jaccard distance
function dist = jaccard_distance(X)
  dist = pdist(X, 'jaccard');
end

% Function to compute Cosine distance
function dist = cosine_distance(X)
  dist = pdist(X, 'cosine');
end

% Calculate pairwise Jaccard distances
jaccard_distances = jaccard_distance(first_two_rows);

% Perform hierarchical clustering with single linkage using Jaccard distance
Z_single_jaccard = linkage(jaccard_distances, 'single');

% Visualize the dendrogram for Jaccard distance with single linkage
figure;
dendrogram(Z_single_jaccard);
title('Hierarchical Clustering Dendrogram (Jaccard Distance)');
xlabel('Emails');
ylabel('Distance (Jaccard)');

% Calculate pairwise Cosine distances
cosine_distances = cosine_distance(first_thousand_rows);

% Perform hierarchical clustering with single linkage using Cosine distance
Z_single_cosine = linkage(cosine_distances, 'single');

% Visualize the dendrogram for Cosine distance with single linkage
figure;
dendrogram(Z_single_cosine);
title('Hierarchical Clustering Dendrogram (Single Linkage, Cosine Distance)');
xlabel('Words');
ylabel('Cosine Distance');

% Calculate pairwise Cosine distances for all data
cosine_distances_all = cosine_distance(all_data);

% Perform hierarchical clustering with average linkage using Cosine distance
Z_average_cosine = linkage(cosine_distances_all, 'average');

% Visualize the dendrogram for Cosine distance with average linkage
figure;
dendrogram(Z_average_cosine);
title('Hierarchical Clustering Dendrogram (Average Linkage, Cosine Distance)');
xlabel('Words');
ylabel('Cosine Distance');

