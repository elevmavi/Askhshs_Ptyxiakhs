function X_cleaned = delNaNsRows(X)
    % Remove rows containing NaN values from an array.
    %
    % Parameters:
    % X: Input array.
    %
    % Returns:
    % Array with NaN-containing rows removed.

    % Check for rows with NaN values
    rows_with_nan = any(isnan(X), 2);

    % Remove rows with NaN values
    X_cleaned = X(~rows_with_nan, :);
end

% Create a sample array with NaN values
X = [1, 2, 3;
     4, NaN, 6;
     7, 8, 9];

% Remove rows with NaN values
X_cleaned = delNaNsRows(X);
disp('Array after removing rows with NaN values:');
disp(X_cleaned);

% Calculate correlation coefficients after removing NaN-containing rows
C = corr(delNaNsRows(X));
disp('Correlation coefficients after removing NaN-containing rows:');
disp(C);

