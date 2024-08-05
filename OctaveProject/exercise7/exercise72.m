pkg load statistics;

function frequent_itemsets = apriori(transactions, min_support)
    % Initialize
    num_transactions = size(transactions, 1);
    num_items = size(transactions, 2);

    % Generate frequent itemsets
    frequent_itemsets = {};
    itemset_size = 1;

    % Generate 1-itemsets
    itemsets = find_itemsets(transactions, itemset_size, min_support);

    while ~isempty(itemsets)
        frequent_itemsets{end + 1} = itemsets;
        itemset_size += 1;
        itemsets = generate_candidates(transactions, itemsets, itemset_size, min_support);
    end

    % Display the frequent itemsets
    disp("Frequent itemsets:");
    for i = 1:length(frequent_itemsets)
        disp(["Size " num2str(i) " itemsets:"]);
        disp(frequent_itemsets{i});
    end
end

function itemsets = find_itemsets(transactions, size1, min_support)
    num_transactions = size(transactions, 1);
    num_items = size(transactions, 2);

    itemsets = {};
    itemset_count = zeros(num_items, 1);

    if size1 == 1
        % Count single item occurrences
        for i = 1:num_items
            itemset_count(i) = sum(transactions(:, i));
        end
    else
        % Count larger itemset occurrences
        num_itemsets = size(transactions, 2);
        for i = 1:num_itemsets
            for j = i+1:num_itemsets
                itemset = [i, j];
                itemset_count(i) = sum(all(transactions(:, itemset), 2));
            end
        end
    end

    % Filter itemsets by support
    for i = 1:num_items
        support = itemset_count(i) / num_transactions;
        if support >= min_support
            itemsets{end + 1} = i; % Store itemsets meeting the minimum support
        end
    end
end

function candidates = generate_candidates(transactions, itemsets, size1, min_support)
    num_items = size(transactions, 2);
    candidates = {};
    % Generate candidate itemsets by combining frequent itemsets
    if size1 > 1
        for i = 1:length(itemsets)
            for j = i+1:length(itemsets)
                itemset1 = itemsets{i};
                itemset2 = itemsets{j};
                candidate = union(itemset1, itemset2);
                if length(candidate) == size1
                    candidates{end + 1} = candidate;
                end
            end
        end
    end
end

% Load the Karate dataset
Karate = load('resources/data/Karate.mat', 'Karate');
Karate = Karate.Karate; % Extract the matrix

% Define parameters
min_support = 0.1;

% Run the Apriori algorithm
frequent_itemsets = apriori(Karate, min_support);

% Display the results
disp("Frequent itemsets:");
for i = 1:length(frequent_itemsets)
    disp(["Size " num2str(i) " itemsets:"]);
    disp(frequent_itemsets{i});
end

% Visualize the adjacency matrix
figure;
spy(Karate, 5);
title('Karate Club Adjacency Matrix');

