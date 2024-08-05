pkg load io;
pkg load odbc; % Load the ODBC package

function data = etl_accdb_file(relative_path, table_name)
  % Load data from an Access database file (.accdb) and fill NaN values with 0

  try
    % Get the current working directory
    current_directory = pwd;

    % Combine the current directory with the relative path
    full_path = fullfile(current_directory, relative_path);
    % Display the full (absolute) path
    disp(full_path);

    % Construct connection string for Access
    conn_str = sprintf('DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;', full_path);

    % Establish connection
    conn = odbc(conn_str, '', '');

    if conn == -1
      error('Failed to connect to the database.');
    endif

    % Execute query to get all data from the specified table
    query = sprintf('SELECT * FROM %s;', table_name);
    data = odbc_exec(conn, query);

    % Fetch data
    data = fetch(data);

    % Rename columns numerically
    col_names = cell(1, size(data, 2));
    for i = 1:size(data, 2)
      col_names{i} = num2str(i - 1);
    end
    data = cell2table(data, 'VariableNames', col_names);

    % Transpose the DataFrame
    data = data';

    % Fill NaN values with 0
    data{isnan(data)} = 0;

    % Close connection
    odbc_close(conn);
  catch err
    error('Error reading .accdb file: %s', err.message);
  end
end

% Load files
xV1 = load('resources/data/xV1.mat').xV1;
xV2 = dlmread('resources/data/xV2.txt', '\t');
xV3 = xlsread('resources/data/xV3.xls')'; %Transpose matrix to keep same format
%xV4 = etl_accdb_file('resources/data/xV4DB.accdb', "xV4");

% Concatenate data frames
%xV = [xV1; xV2; xV3; xV4];
xV = [xV1; xV2; xV3];
disp("Concatenated data frame has NaN values: " + num2str(any(isnan(xV(:)))));

% Plot scatter diagram for the first two columns
figure(1);
scatter(xV(:, 1), xV(:, 2), 'filled');
title('xV Columns 0 vs 1');

% Plot scatter diagrams for each column pair
figure(2);
for i = 1:12
    subplot(3, 4, i);
    scatter(xV(:, i), xV(:, i + 1), 'filled');
    title(['xV Columns ', num2str(i), ' vs ', num2str(i + 1)]);
    xlabel(['Column ', num2str(i)]);
    ylabel(['Column ', num2str(i + 1)]);
end

% Create xVa, xVb, xVc
xVa = [xV1(1:50, :); xV2(1:50, :); xV3(1:50, :)];
%xVa = [xV1(1:50, :); xV2(1:50, :); xV3(1:50, :); xV4(1:50, :)];
xVb = [xV1(51:100, :); xV2(51:100, :); xV3(51:100, :)];
%xVb = [xV1(51:100, :); xV2(51:100, :); xV3(51:100, :); xV4(51:100, :)];
xVc = [xV1(101:150, :); xV2(101:150, :); xV3(101:150, :)];
%xVc = [xV1(101:150, :); xV2(101:150, :); xV3(101:150, :); xV4(101:150, :)];

% Concatenate xVa, xVb, xVc
xVd = [xVa; xVb; xVc];

% Plot for xVd
figure(3);
for i = 1:100
    subplot(10, 10, i);
    x = [xVd(1:200, i); xVd(201:400, i); xVd(401:450, i)];
    y = [xVd(1:200, i + 1); xVd(201:400, i + 1); xVd(401:450, i + 1)];
    plot(x, y, '.');
    title(['xVd Plot ', num2str(i)]);
end

% Scatter diagrams for xV2
figure(4);
for i = 1:12
    subplot(3, 4, i);
    scatter(xV2(:, i), xV2(:, i + 1), 'filled');
    title(['xV2 Columns ', num2str(i), ' vs ', num2str(i + 1)]);
    xlabel(['Column ', num2str(i)]);
    ylabel(['Column ', num2str(i + 1)]);
end

% Plot for xV2
figure(5);
for i = 1:100
    subplot(10, 10, i);
    x = [xV2(1:50, i); xV2(51:100, i); xV2(101:150, i)];
    y = [xV2(1:50, i + 1); xV2(51:100, i + 1); xV2(101:150, i + 1)];
    plot(x, y, '.');
    title(['xV2 Plot ', num2str(i)]);
end


