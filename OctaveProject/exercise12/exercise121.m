pkg load io; % For loading .mat files
pkg load statistics; % For statistics functions
pkg load image; % For colormap functions (optional for heatmap visualization)

% Load data
data = load('resources/data/xV600x470.mat');
initidx = data.xV1(:, 1);
xV = data.xV1(:, 2:end);

% Remove columns with NaN values
xV = xV(:, ~any(isnan(xV), 1));

% Calculate Fisher score
class_means = mean(xV(initidx == 1, :), 1);
overall_mean = mean(xV, 1);
class_variance = var(xV(initidx == 1, :), 0, 1);
overall_variance = var(xV, 0, 1);
fisher_score = ((class_means - overall_mean) .^ 2) ./ (class_variance + overall_variance);

% Sort features based on Fisher score
[~, sorted_indices] = sort(fisher_score, 'descend');

% Plot heatmap using imagesc
rho = corrcoef(xV(:, 1:min(50, size(xV, 2)))); % Only the first 50 features or fewer if less

% Create a new figure
figure;
imagesc(rho); % Plot the correlation matrix
colorbar; % Add a color bar to the side
title('Correlation Heatmap');
xlabel('Features');
ylabel('Features');

% Adjust color limits if needed
caxis([-1 1]); % Set color limits for better visualization

% Find and print highly correlated features
threshold = 0.5;
abs_rho = abs(rho);
highly_correlated = find(abs_rho >= threshold);
num_corr = length(highly_correlated);
correlated_pairs = [];

for i = 1:num_corr
    [row, col] = ind2sub(size(rho), highly_correlated(i));
    if row < col
        correlated_pairs = [correlated_pairs; row, col, rho(row, col)];
    endif
endfor

disp('Highly correlated features:');
disp(correlated_pairs);

% Print features with high Fisher score
num_features_to_keep = 50;
selected_features = sorted_indices(1:num_features_to_keep);
disp('Selected features based on Fisher score:');
disp(selected_features);

