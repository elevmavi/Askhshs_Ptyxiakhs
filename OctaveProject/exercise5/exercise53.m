pkg load statistics; % For statistical functions

% Manual ROC curve calculation
function [fpr, tpr, thresholds] = calculate_roc(y_true, y_scores)
    [~, sorted_indices] = sort(y_scores, 'descend');
    y_true_sorted = y_true(sorted_indices);

    thresholds = unique(y_scores);
    num_thresholds = length(thresholds);

    tpr = zeros(num_thresholds, 1);
    fpr = zeros(num_thresholds, 1);

    for i = 1:num_thresholds
        threshold = thresholds(i);
        y_pred = (y_scores >= threshold);

        tp = sum((y_true == 1) & (y_pred == 1));
        fp = sum((y_true == 0) & (y_pred == 1));
        fn = sum((y_true == 1) & (y_pred == 0));
        tn = sum((y_true == 0) & (y_pred == 0));

        tpr(i) = tp / (tp + fn);
        fpr(i) = fp / (fp + tn);
    end

    % Append the (0,0) and (1,1) points
    tpr = [0; tpr; 1];
    fpr = [0; fpr; 1];
    thresholds = [max(thresholds); thresholds; min(thresholds)];
end

% Function to perform k-NN classification and return probabilities
function y_prob = knn_classify(X_train, y_train, X_test, k)
    n_test = size(X_test, 1);
    y_prob = zeros(n_test, 1);

    for i = 1:n_test
        % Compute distances from the i-th test point to all training points
        dists = compute_distances(X_train, X_test(i, :));

        % Get the indices of the k nearest neighbors
        [~, idx] = sort(dists);
        nearest_idx = idx(1:k);

        % Compute probabilities: fraction of neighbors in class 1
        y_prob(i) = sum(y_train(nearest_idx)) / k;
    end
end

% Load dataset (assuming it is saved in a CSV format or similar)
% You'll need to adapt this part based on your actual data format
data = dlmread('resources/data/heart_disease.csv', '\t'); % Replace with your dataset path
X = data(:, 1:end); % Features
y = data(:, end); % Targets

% Convert continuous target to binary class labels (example threshold)
threshold = median(y); % Example threshold for binary classification
y_binary = y <= threshold; % Binary class labels (0 or 1)

% Handle missing values (using median imputation)
for i = 1:size(X, 2)
  col = X(:, i);
  missing_indices = isnan(col);
  col(missing_indices) = median(col(~missing_indices));
  X(:, i) = col;
end

% Subsample the data to speed up processing
% Define the size of the subset
subset_fraction = 0.1;  % Use 10% of the data
n = size(X, 1);
subset_size = round(subset_fraction * n);

% Randomly select a subset of the data
indices = randperm(n, subset_size);

X_subset = X(indices, :);
y_subset = y_binary(indices);

% Split the subset data into train and test sets (80% train, 20% test)
train_size = round(0.8 * subset_size);
indices = randperm(subset_size);

X_train = X_subset(indices(1:train_size), :);
y_train = y_subset(indices(1:train_size));
X_test = X_subset(indices(train_size+1:end), :);
y_test = y_subset(indices(train_size+1:end));

% List of k values to test
k_values = [1, 3, 5, 7, 9];

for k = k_values
    % Initialize kNN and predict
    % Octave doesn't have a direct kNN classifier, so use knnsearch for predictions
    y_pred = knn_classify(X_train, y_train, X_test, k);

    % Ensure y_test and y_pred are column vectors
    y_pred = logical(y_pred);

    % Evaluate performance
    % Calculate accuracy
    accuracy = sum(y_test == y_pred) / length(y_test);

    % Calculate confusion matrix
    conf_matrix = confusionmat(y_test, y_pred);

    % Calculate precision, recall, and F1 score
    tp = conf_matrix(2, 2);
    fp = conf_matrix(1, 2);
    fn = conf_matrix(2, 1);
    tn = conf_matrix(1, 1);

    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * (precision * recall) / (precision + recall);

    % Compute ROC curve
    [fpr, tpr, thresholds] = calculate_roc(y_test, y_pred);
    auc_roc = trapz(fpr, tpr);

    % Compute Precision-Recall curve (manually or using an external method)
    % Octave does not provide a built-in PR curve function. You might need to implement it or use a package.

    printf("Performance for k = %d with all features:\n", k);
    printf("Accuracy: %f\n", accuracy);
    printf("Confusion Matrix:\n");
    disp(conf_matrix);
    printf("Precision: %f\n", precision);
    printf("Recall: %f\n", recall);
    printf("F1 Score: %f\n", f1);
    printf("AUC-ROC: %f\n", auc_roc);
    printf("\n");
end

