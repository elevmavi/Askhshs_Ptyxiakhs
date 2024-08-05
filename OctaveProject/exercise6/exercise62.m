pkg load statistics;
pkg load io;

function [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size)
  % Randomly split data into training and test sets
  num_samples = size(X, 1);
  indices = randperm(num_samples);
  num_test = round(test_size * num_samples);
  test_indices = indices(1:num_test);
  train_indices = indices(num_test+1:end);

  X_train = X(train_indices, :);
  X_test = X(test_indices, :);
  y_train = y(train_indices);
  y_test = y(test_indices);
endfunction

function nb_model = train_naive_bayes(X_train, y_train)
  % Train Gaussian Naive Bayes model
  nb_model = struct();
  [num_samples, num_features] = size(X_train);
  classes = unique(y_train);
  num_classes = length(classes);

  % Initialize means and stds
  nb_model.means = zeros(num_classes, num_features);
  nb_model.stds = zeros(num_classes, num_features);
  nb_model.priors = zeros(num_classes, 1);

  % Calculate means, stds and priors for each class
  for i = 1:num_classes
    class_idx = (y_train == classes(i));
    nb_model.means(i, :) = mean(X_train(class_idx, :), 1);
    nb_model.stds(i, :) = std(X_train(class_idx, :), 0, 1);
    nb_model.priors(i) = sum(class_idx) / num_samples;
  endfor
endfunction

function [y_pred, probs] = predict_naive_bayes(nb_model, X_test)
  % Predict labels and posterior probabilities using Naive Bayes
  num_samples = size(X_test, 1);
  num_classes = length(nb_model.priors);
  num_features = size(X_test, 2); % Define num_features here
  probs = zeros(num_samples, num_classes);

  % Compute class probabilities
  for i = 1:num_classes
    mu = nb_model.means(i, :);
    sigma = nb_model.stds(i, :);
    prior = nb_model.priors(i);

    % Compute log-probabilities
    log_prob = -0.5 * sum(((X_test - mu) ./ sigma) .^ 2, 2)
                - num_features * log(sigma) - num_features * log(sqrt(2 * pi));
    log_prob = log_prob + log(prior);
    probs(:, i) = exp(log_prob);
  endfor

  % Normalize probabilities
  probs = probs ./ sum(probs, 2);

  % Predict classes
  [~, y_pred] = max(probs, [], 2);
endfunction


function [conf_matrix, accuracy, error_rate, sensitivity, specificity, y_pred] = evaluate_classifier(X_train, X_test, y_train, y_test)
  % Train Naive Bayes model
  nb_model = train_naive_bayes(X_train, y_train);

  % Predict test data
  [y_pred, probs] = predict_naive_bayes(nb_model, X_test);

  % Compute confusion matrix
  conf_matrix = confusionmat(y_test, y_pred);

  % Compute accuracy
  accuracy = sum(y_test == y_pred) / length(y_test);
  % Compute error rate
  error_rate = 1 - accuracy;

  % Compute sensitivity (macro average recall)
  sensitivity = compute_recall(y_test, y_pred);

  % Compute specificity
  num_classes = size(conf_matrix, 1);
  specificities = zeros(num_classes, 1);
  for i = 1:num_classes
    tn = sum(conf_matrix(:)) - sum(conf_matrix(i, :)) - sum(conf_matrix(:, i)) + conf_matrix(i, i);
    fp = sum(conf_matrix(:, i)) - conf_matrix(i, i);
    specificities(i) = tn / (tn + fp);
  endfor
  specificity = mean(specificities);
endfunction

function recall = compute_recall(y_true, y_pred)
  % Compute recall for each class
  classes = unique(y_true);
  num_classes = length(classes);
  recall = zeros(num_classes, 1);

  for i = 1:num_classes
    class = classes(i);
    true_positives = sum((y_true == class) & (y_pred == class));
    false_negatives = sum((y_true == class) & (y_pred != class));

    recall(i) = true_positives / (true_positives + false_negatives);
  endfor

  % Return macro-average recall
  recall = mean(recall);
endfunction

% Load data
data = load('resources/data/xV2.mat');
X = data.xV(:, 1:2);  % Select the first two columns as features
y = data.xV(:, 6);    % Select the sixth column as target class

% Split data into training and test sets (70% train, 30% test)
[X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.3);

% Evaluate classifier
[conf_matrix, accuracy, error_rate, sensitivity, specificity, y_pred] = evaluate_classifier(X_train, X_test, y_train, y_test);

printf("Accuracy: %f\n", accuracy);
printf("Confusion Matrix:\n");
disp(conf_matrix);
printf("Error rate: %f\n", error_rate);
printf("Sensitivity: %f\n", sensitivity);
printf("Specificity: %f\n", specificity);

% Plot predicted vs. actual classes for the test set
figure;
scatter(X_test(:, 1), X_test(:, 2), 50, y_pred, 'filled');
hold on;
scatter(X_test(:, 1), X_test(:, 2), 100, y_test, 'x');
title('Naive Bayes Classifier: Predicted vs. Actual Classes');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Predicted', 'Actual');
hold off;

% Repeat the process 10 times with random subsampling
num_iterations = 10;
accuracies = zeros(num_iterations, 1);
conf_matrices = cell(num_iterations, 1);
error_rates = zeros(num_iterations, 1);
sensitivities = zeros(num_iterations, 1);
specificities = zeros(num_iterations, 1);
posterior_probs = cell(num_iterations, 1);


for i = 1:num_iterations
  [X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.3);

   % Train the Naive Bayes model
  nb_model = train_naive_bayes(X_train, y_train);

  [conf_matrix, accuracy, error_rate, sensitivity, specificity, y_pred] = evaluate_classifier(X_train, X_test, y_train, y_test);

  accuracies(i) = accuracy;
  conf_matrices{i} = conf_matrix;
  error_rates(i) = error_rate;
  sensitivities(i) = sensitivity;
  specificities(i) = specificity;
  posterior_probs{i} = predict_naive_bayes(nb_model, X_test);

  % Plot predicted vs. actual classes for the test set
  figure;
  scatter(X_test(:, 1), X_test(:, 2), 50, y_pred(i), 'filled');
  hold on;
  scatter(X_test(:, 1), X_test(:, 2), 100, y_test, 'x');
  title('Naive Bayes Classifier: Predicted vs. Actual Classes');
  xlabel('Feature 1');
  ylabel('Feature 2');
  legend('Predicted', 'Actual');
  hold off;
endfor

% Compute mean accuracy and confusion matrix across iterations
mean_accuracy = mean(accuracies);
mean_conf_matrix = mean(cat(3, conf_matrices{:}), 3);
mean_error_rates = mean(error_rates);
mean_sensitivities = mean(sensitivities);
mean_specificity = mean(specificities);
mean_posterior_probs = mean(cat(3, posterior_probs{:}), 3);

printf("Mean Accuracy: %f\n", mean_accuracy);
printf("Mean Confusion Matrix:\n");
disp(mean_conf_matrix);
printf("Mean Error rates: %f\n", mean_error_rates);
printf("Mean Sensitivities: %f\n", mean_sensitivities);
printf("Mean Specificities: %f\n", mean_specificity);
printf("Mean Posterior Probs:\n");
disp(mean_posterior_probs);

