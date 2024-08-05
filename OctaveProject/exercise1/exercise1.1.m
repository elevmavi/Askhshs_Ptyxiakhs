# Load the statistics package for nansum
pkg load statistics;

# Create a 3x3 magic square with float data type
a = [8, 1, 6;
     3, 5, 7;
     4, 9, 2];  # In Octave, matrices are created with semicolons separating rows

# Calculate the sum of all elements in the matrix along the first axis (columns)
sum_a = sum(a, 1);  # sum(a, 1) calculates the sum along columns
disp("Sum of all elements in a:");
disp(sum_a);

# Transpose the matrix and then calculate the sum (equivalent to sum(a')')
sum_a_transposed = sum(a', 1);  # Transpose the matrix and then sum along columns
disp("Sum of all elements in a transposed:");
disp(sum_a_transposed);

# Set element at (2,2) to NaN
a(2, 2) = NaN;

# Calculate the sum of all elements in the modified matrix (ignoring NaN values)
sum_a_modified = nansum(a, 1);  # Use nansum to ignore NaN values
disp("Sum of all elements in a (with NaN):");
disp(sum_a_modified);

