pkg load statistics; % Load the statistics package for distance functions

% Load the data
iris = dlmread('resources/data/iris.txt', ',');

% Extract the data
X1 = iris(:, 1:4);

% Select the first observation as the reference point
reference_observation1 = X1(1, :);

% Compute Euclidean distances from the reference observation to all other observations
euclidean_d_to_reference1 = sqrt(sum((X1 - reference_observation1).^2, 2));

% Compute Cityblock distances from the reference observation to all other observations
cityblock_d_to_reference = sum(abs(X1 - reference_observation1), 2);

% Plot Euclidean distances
figure;
plot(euclidean_d_to_reference1, 'o-', 'Color', 'b', 'DisplayName', 'Euclidean distance from 1st Observation');
title('Euclidean Distance from First Observation to Other Observations');
xlabel('Observation Index');
ylabel('Distance to First Observation');
grid on;
legend;

% Plot Cityblock distances
figure;
plot(cityblock_d_to_reference, 's-', 'Color', 'g', 'DisplayName', 'Cityblock distance from 1st Observation');
title('Cityblock Distance from First Observation to Other Observations');
xlabel('Observation Index');
ylabel('Distance to First Observation');
grid on;
legend;

% Extract the second dataset
X2 = iris(:, 1:2);
reference_observation2 = X2(1, :);

% Compute Euclidean distances for the second dataset
euclidean_d_to_reference2 = sqrt(sum((X2 - reference_observation2).^2, 2));

% Plot Euclidean distances for both datasets
figure;
plot(euclidean_d_to_reference1, 'o-', 'Color', 'b', 'DisplayName', 'Euclidean distance 0-4');
hold on;
plot(euclidean_d_to_reference2, 's-', 'Color', 'g', 'DisplayName', 'Euclidean distance 0-2');
xlabel('Observation Index');
ylabel('Distance Value');
title('Euclidean Distance Metrics');
grid on;
legend;
hold off;

