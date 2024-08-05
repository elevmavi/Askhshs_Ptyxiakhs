% Load the data
data = load('resources/data/mydata.mat');
X = data.X;

% Extract columns for plotting
column1 = X(:, 1);
column2 = X(:, 2);

% Set DBSCAN parameters
epsilon = 0.5;
MinPts = 15;

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
IDX = dbscan(X, epsilon, MinPts);

% Plotting original data
figure(1);
scatter(column1, column2);
title('Original Data');
xlabel('Feature 1');
ylabel('Feature 2');

% Plotting clustering results
figure(2);
scatter(column1, column2, 10, IDX, 'filled'); % '10' specifies the marker size
title(['DBSCAN Clustering (epsilon = ', num2str(epsilon), ', MinPts = ', num2str(MinPts), ')']);
xlabel('Feature 1');
ylabel('Feature 2');
colorbar('southoutside'); % To match Python's colorbar positioning

