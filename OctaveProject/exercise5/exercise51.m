pkg load statistics;  % Load the statistics package for kNN and other functions
pkg load image;  % For confusion matrix heatmap

function metrics = calculate_performance_metrics(conf_matrix)
    num_classes = rows(conf_matrix);
    metrics = struct();

    for i = 1:num_classes
        if (TN + FP) > 0
            specificity = TN / (TN + FP);
        else
            specificity = 0;
        end

        if (TP + FP) > 0
            precision = TP / (TP + FP);
        else
            precision = 0;
        end

        TP = conf_matrix(i, i);
        FP = sum(conf_matrix(:, i)) - TP;
        FN = sum(conf_matrix(i, :)) - TP;
        TN = sum(conf_matrix(:)) - (TP + FP + FN);

        % Calculate metrics with conditional checks
        if (TP + FN) > 0
            sensitivity = TP / (TP + FN);
        else
            sensitivity = 0;
        end

        if (TN + FN) > 0
            npv = TN / (TN + FN);
        else
            npv = 0;
        end

        if (FP + TN) > 0
            fpr = FP / (FP + TN);
        else
            fpr = 0;
        end

        if (FN + TP) > 0
            fnr = FN / (FN + TP);
        else
            fnr = 0;
        end

        if fpr > 0
            lrp = sensitivity / fpr;
        else
            lrp = Inf;
        end

        if specificity > 0
            lrn = fnr / specificity;
        else
            lrn = Inf;
        end

        era = (fpr + fnr) / 2;

        if (precision + sensitivity) > 0
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);
        else
            f1_score = 0;
        end

        metrics.(sprintf('Class %d', i)) = struct('TP', TP, 'FP', FP, 'FN', FN, 'TN', TN,
            'Accuracy', (TP + TN) / sum(conf_matrix(:)), 'Error Rate', (FP + FN) / sum(conf_matrix(:)),
            'Sensitivity (Recall)', sensitivity, 'Specificity', specificity, 'Precision', precision,
            'NPV (Negative Predictive Value)', npv, 'FPR (False Positive Rate)', fpr,
            'FNR (False Negative Rate)', fnr, 'LR+ (Positive Likelihood Ratio)', lrp,
            'LR- (Negative Likelihood Ratio)', lrn, 'ERA (Error Rate Average)', era,
            'F1 Score', f1_score);
    end
end


% Load iris data
data = load('C:/Users/User/OctaveProject/OctaveProject/resources/data/iris.txt');  % Replace with correct data loading method if necessary

% Extracting the required dimensions from the matrix
X = data(:, [3, 4]);  % Selecting the 3rd and 4th columns (1-based indexing)
Y = data(:, 5);  % Selecting the 5th column (1-based indexing)

% Splitting the dataset into training (X1) and test (X2) sets
X1 = X([1:40, 51:90, 101:140], :);  % Training set
X2 = X([41:50, 91:100, 141:150], :);  % Test set

% Selecting the corresponding classes for training and test sets
c1 = Y([1:40, 51:90, 101:140]);  % Classes for training set
c2 = Y([41:50, 91:100, 141:150]);  % Classes for test set

% Applying the k-Nearest Neighbors (kNN) algorithm
k = 3;  % Number of nearest neighbors to consider

% Fit the classifier with the training data and corresponding classes
mdl = fitcknn(X1, c1, 'NumNeighbors', k, 'Distance', 'euclidean');

% Predicting classes for the test data
predicted_classes = predict(mdl, X2);
predicted_classes = cellfun(@str2double, predicted_classes);

% Plotting kNN classified classes using scatter
indices = 1:length(predicted_classes);
figure;
scatter(indices, predicted_classes, 'r', 'filled');
hold on;
scatter(indices, c2, 'b');
legend('kNN class', 'Original Class', 'location', 'northwest');
xlabel('Sample Index');
ylabel('Class');
title('kNN Classification vs. Original Classes');
hold off;

% Calculate confusion matrix
conf_matrix = confusionmat(c2, predicted_classes);

% Display the confusion matrix as a heatmap
figure;
imagesc(conf_matrix);
colormap('cool');
colorbar;
xticks(1:3);
yticks(1:3);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix');
for i = 1:size(conf_matrix, 1)
    for j = 1:size(conf_matrix, 2)
        text(j, i, num2str(conf_matrix(i, j)), 'HorizontalAlignment', 'center', 'Color', 'black');
    end
end

% Calculate performance metrics
metrics = calculate_performance_metrics(conf_matrix);

% Display the performance metrics
fields = fieldnames(metrics);
for i = 1:numel(fields)
    class_label = fields{i};
    disp(sprintf('Class %s:', class_label));
    class_metrics = metrics.(class_label);
    metrics_fields = fieldnames(class_metrics);
    for j = 1:numel(metrics_fields)
        metric_name = metrics_fields{j};
        value = class_metrics.(metric_name);
        if isnumeric(value)
            disp(sprintf('%s: %.4f', metric_name, value));
        else
            disp(sprintf('%s: %s', metric_name, value));
        end
    end
    disp(' ');
end

% Cross-validation
num_folds = 10;
cv = cvpartition(size(X, 1), 'KFold', num_folds);
cv_scores = zeros(num_folds, 1);

for i = 1:num_folds
    train_idx = training(cv, i);
    test_idx = test(cv, i);

    X_train = X(train_idx, :);
    X_test = X(test_idx, :);
    c1_train = Y(train_idx);
    c2_test = Y(test_idx);

    mdl = fitcknn(X_train, c1_train, 'NumNeighbors', k, 'Distance', 'euclidean');
    predicted_classes = predict(mdl, X_test);
    predicted_classes = cellfun(@str2double, predicted_classes);

    conf_matrix = confusionmat(c2_test, predicted_classes);
    metrics = calculate_performance_metrics(conf_matrix);
    accuracy = 1 - (sum(conf_matrix(:)) - trace(conf_matrix)) / sum(conf_matrix(:));
    cv_scores(i) = accuracy;
end

mean_error_rate = 1 - mean(cv_scores);
disp(sprintf('Mean Error Rate across %d folds: %.4f', num_folds, mean_error_rate));

% Experiment with different dimensions and k values
dimensions_to_try = {
    [1], [2], [3], [4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4], [1, 2, 3, 4]
};

neighbors_to_try = [1, 2, 5, 15];
results = containers.Map();

for d = 1:numel(dimensions_to_try)
    dimensions = dimensions_to_try{d};
    X_subset = data(:, dimensions);  % Subset of X based on selected dimensions

    for k = neighbors_to_try
        mdl = fitcknn(X_subset, Y, 'NumNeighbors', k, 'Distance', 'euclidean');

        cv = cvpartition(size(X_subset, 1), 'KFold', num_folds);
        cv_scores = zeros(num_folds, 1);

        for i = 1:num_folds
            train_idx = training(cv, i);
            test_idx = test(cv, i);

            X_train = X_subset(train_idx, :);
            X_test = X_subset(test_idx, :);
            c1_train = Y(train_idx);
            c2_test = Y(test_idx);
        end

        mdl = fitcknn(X_train, c1_train, 'NumNeighbors', k, 'Distance', 'euclidean');
        predicted_classes = predict(mdl, X_test);
        predicted_classes = cellfun(@str2double, predicted_classes);

        conf_matrix = confusionmat(c2_test, predicted_classes);
        accuracy = 1 - (sum(conf_matrix(:)) - trace(conf_matrix)) / sum(conf_matrix(:));
        cv_scores(i) = accuracy;
        mean_error_rate = 1 - mean(cv_scores);
        results(num2str([dimensions, k])) = mean_error_rate;

        % Plotting kNN classified classes using scatter
        indices = 1:length(predicted_classes);
        figure;
        scatter(indices, predicted_classes, 'r', 'filled');
        hold on;
        scatter(indices, c2_test, 'b');
        legend('kNN class', 'Original Class', 'location', 'northwest');
        xlabel('Sample Index');
        ylabel('Class');
        title('kNN Classification vs. Original Classes');
        hold off;

        % Display the confusion matrix as a heatmap
        figure;
        imagesc(conf_matrix);
        colormap('cool');
        colorbar;
        xticks(1:3);
        yticks(1:3);
        xlabel('Predicted Class');
        ylabel('True Class');
        title('Confusion Matrix');
        for i = 1:size(conf_matrix, 1)
            for j = 1:size(conf_matrix, 2)
                text(j, i, num2str(conf_matrix(i, j)), 'HorizontalAlignment', 'center', 'Color', 'black');
            end
        end

        % Calculate performance metrics
        all_data_metrics = calculate_performance_metrics(conf_matrix);

        % Display the performance metrics
        fields = fieldnames(all_data_metrics);
        for i = 1:numel(fields)
            class_label = fields{i};
            disp(sprintf('Class %s:', class_label));
            class_metrics = all_data_metrics.(class_label);
            metrics_fields = fieldnames(class_metrics);
            for j = 1:numel(metrics_fields)
                metric_name = metrics_fields{j};
                value = class_metrics.(metric_name);
                if isnumeric(value)
                    disp(sprintf('%s: %.4f', metric_name, value));
                else
                    disp(sprintf('%s: %s', metric_name, value));
                end
            end
            disp(' ');
        end

        disp(sprintf('Mean Error Rate across %d folds: %.4f', num_folds, mean_error_rate));
    end
end

% Print results
keys = keys(results);
for i = 1:numel(keys)
    config = keys{i};
    error_rate = results(config);
    disp(sprintf('Configuration: %s - Mean Error Rate: %.4f', config, error_rate));
end

