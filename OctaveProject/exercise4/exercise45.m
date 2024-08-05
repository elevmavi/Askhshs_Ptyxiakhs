% File: exercise45.m

% Load data
data = load('resources/data/xV.mat');
xV = data.xV;
feature1 = xV(:, [1, 2]);

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

% Perform DBSCAN clustering
epsilon = 0.3;
min_samples = 50;
labels = dbscan(feature1, epsilon, min_samples);

% Plot results
figure;
scatter(feature1(:, 1), feature1(:, 2), 50, labels, 'filled');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
title('DBSCAN Clustering (eps = 0.3, minSamples = 50)');
colorbar;

% Repeat for other parameters
epsilon = 0.2;
min_samples = 4;
labels = dbscan(feature1, epsilon, min_samples);

figure;
scatter(feature1(:, 1), feature1(:, 2), 50, labels, 'filled');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
title('DBSCAN Clustering (eps = 0.2, minSamples = 4)');
colorbar;

epsilon = 0.7;
min_samples = 10;
labels = dbscan(feature1, epsilon, min_samples);

figure;
scatter(feature1(:, 1), feature1(:, 2), 50, labels, 'filled');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
title('DBSCAN Clustering (eps = 0.7, minSamples = 10)');
colorbar;

