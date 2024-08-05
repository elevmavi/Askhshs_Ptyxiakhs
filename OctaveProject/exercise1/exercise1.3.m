% Given data
t = 1900:10:1990;
p = [75.995, 91.972, 105.711, 123.203, 131.669, 150.697, 179.323, 203.212, 226.505, 249.633];

% Interpolation at a specific point (1975) using linear interpolation
p_interp_1975 = interp1(t, p, 1975);
fprintf('Interpolated value at 1975: %.3f\n', p_interp_1975);

% Define the range of x values for interpolation
x = 1900:2000;  % from 1900 to 2000 inclusive

% Perform nearest neighbor interpolation
[~, indices] = min(abs(bsxfun(@minus, x(:), t)), [], 2);
indices = indices';  % Transpose to match the shape of x
y = p(indices);  % Perform nearest neighbor interpolation

% Plotting
figure;
plot(t, p, 'o', 'DisplayName', 'Original Data');  % Plot original data points
hold on;
plot(x, y, 'DisplayName', 'Interpolated Data (Nearest)', 'LineStyle', '--');  % Plot interpolated data
title('Nearest Neighbor Interpolation');
xlabel('Year');
ylabel('Population (in millions)');
legend show;
grid on;
hold off;

% Given data for cubic spline interpolation
tab = [1950, 150.697;
       1960, 179.323;
       1970, 203.212;
       1980, 226.505;
       1990, 249.633];

% Create a cubic spline interpolation function
xq = linspace(1950, 1990, 100);  % Define query points for interpolation
yq = interp1(tab(:, 1), tab(:, 2), xq, 'spline');  % Cubic spline interpolation

% Interpolation at a specific point (1975) using cubic spline
p_1975 = interp1(tab(:, 1), tab(:, 2), 1975, 'spline');
fprintf('Interpolated value at 1975 (Cubic Spline): %.3f\n', p_1975);

% Plotting
figure;
plot(tab(:, 1), tab(:, 2), 'o', 'DisplayName', 'Original Data');  % Plot original data points
hold on;
plot(xq, yq, 'DisplayName', 'Interpolated Data (Cubic Spline)', 'LineStyle', '--');  % Plot interpolated data
title('Population Interpolation using Cubic Spline');
xlabel('Year');
ylabel('Population (in millions)');
legend show;
grid on;
hold off;

