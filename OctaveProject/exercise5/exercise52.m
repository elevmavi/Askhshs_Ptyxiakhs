pkg load statistics;  # Load the statistics package for machine learning functions

function xV = etl_mat_file_fill_na(file_path, variable_name)
    # Load the .mat file
    data = load(file_path);
    xV = data.(variable_name);
    # Fill missing values with appropriate strategy, e.g., mean of the column
    for i = 1:size(xV, 2)
        xV(isnan(xV(:, i)), i) = mean(xV(~isnan(xV(:, i)), i));
    end
end


pkg load statistics;  # Load the statistics package for machine learning functions

# Load the .mat file and fill missing values (assuming a function `etl_mat_file_fill_na` is defined)
xV = etl_mat_file_fill_na('resources/data/xV.mat', 'xV');

# Split data into features (X) and target (y)
X = xV(:, 1:end-1);  # Features: all columns except the last one
Y = xV(:, end);  # Target: last column

# Convert continuous target to binary class labels (example threshold)
threshold = median(Y);  # Example threshold for binary classification
# Map continuous values to binary classes (0 or 1)
y_binary = Y <= threshold;

# Split data into training and testing sets (70% training, 30% testing)
n = size(X, 1);
idx = randperm(n);
train_idx = idx(1:round(0.7 * n));
test_idx = idx(round(0.7 * n) + 1:end);

X_train = X(train_idx, :);
X_test = X(test_idx, :);
y_train = y_binary(train_idx);
y_test = y_binary(test_idx);

# Define k values
k_values = [1, 3, 5, 20];

# Loop over different k values
for k = k_values
    # Initialize and train kNN classifier
    mdl = fitcknn(X_train, y_train, 'NumNeighbors', k);

    # Predict on the test set
    y_pred = predict(mdl, X_test);
    y_pred = logical(cellfun(@str2double, y_pred));
    # Evaluate performance
    accuracy = mean(y_pred == y_test);
    confusion = confusionmat(y_test, y_pred);
    precision = sum((y_pred == 1) & (y_test == 1)) / sum(y_pred == 1);
    recall = sum((y_pred == 1) & (y_test == 1)) / sum(y_test == 1);
    printf('Performance for k = %d:\n', k);
    printf('Accuracy: %.2f\n', accuracy);
    printf('Confusion Matrix:\n');
    disp(confusion);
    printf('Precision: %.2f\n', precision);
    printf('Recall: %.2f\n', recall);
    printf('\n');
end

# Plotting the classification performance for the first two features
figure;

# Separate the data by class
class_0_idx = y_pred == 0;
class_1_idx = y_pred == 1;

# Plot each class with a different color
scatter(X_test(class_0_idx, 1), X_test(class_0_idx, 2), 'r', 'filled');  # Class 0 in red
hold on;
scatter(X_test(class_1_idx, 1), X_test(class_1_idx, 2), 'b', 'filled');  # Class 1 in blue
hold off;

title('Predicted Classes');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 0', 'Class 1');

# Extract all features (excluding the last column which is assumed to be the target)
X_all = xV(:, 1:end-1);

# Split data into training and testing sets
n_all = size(X_all, 1);
idx_all = randperm(n_all);
train_idx_all = idx_all(1:round(0.7 * n_all));
test_idx_all = idx_all(round(0.7 * n_all) + 1:end);

X_train_all = X_all(train_idx_all, :);
X_test_all = X_all(test_idx_all, :);
y_train_all = y_binary(train_idx_all);
y_test_all = y_binary(test_idx_all);

# Loop over different k values
for k = k_values
    # Initialize and train kNN classifier
    mdl_all = fitcknn(X_train_all, y_train_all, 'NumNeighbors', k);

    # Predict on the test set
    y_pred_all = predict(mdl_all, X_test_all);
    y_pred_all = logical(cellfun(@str2double, y_pred_all));

    # Evaluate performance
    accuracy_all = mean(y_pred_all == y_test_all);
    confusion_all = confusionmat(y_test_all, y_pred_all);
    precision_all = sum((y_pred_all == 1) & (y_test_all == 1)) / sum(y_pred_all == 1);
    recall_all = sum((y_pred_all == 1) & (y_test_all == 1)) / sum(y_test_all == 1);

    printf('Performance for k = %d with all features:\n', k);
    printf('Accuracy: %.2f\n', accuracy_all);
    printf('Confusion Matrix:\n');
    disp(confusion_all);
    printf('Precision: %.2f\n', precision_all);
    printf('Recall: %.2f\n', recall_all);
    printf('\n');
end

# Plotting the classification performance for the first two features
figure;

# Separate the data by class
class_0_idx = y_pred_all == 0;
class_1_idx = y_pred_all == 1;

# Plot each class with a different color
scatter(X_test_all(class_0_idx, 1), X_test_all(class_0_idx, 2), 'r', 'filled');  # Class 0 in red
hold on;
scatter(X_test_all(class_1_idx, 1), X_test_all(class_1_idx, 2), 'b', 'filled');  # Class 1 in blue
hold off;

title('Predicted Classes');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 0', 'Class 1');


