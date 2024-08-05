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

% Sample transaction data
data = {
    'Milk', 'Bread', 'Butter', 'Beer', 'Diapers';
    true, true, false, false, false;
    false, false, true, false, false;
    false, true, true, true, true;
    true, true, true, false, false;
    false, true, false, false, false
};

% Convert data to logical matrix
headers = data(1, 1:end);
transactions = cell2mat(data(2:end, :));
transactions = logical(transactions);

% Convert data to logical matrix for processing
disp("Transactions:");
disp(transactions);

% Apply Apriori algorithm to find frequent itemsets
min_support = 0.3;
frequent_itemsets = apriori(transactions, min_support);

% Calculate support, confidence, and lift for item pairs
num_items = size(transactions, 2);
support = zeros(num_items, num_items);
confidence = zeros(num_items, num_items);
lift = zeros(num_items, num_items);

for i = 1:num_items
    for j = 1:num_items
        if i != j
            sup_both = sum(transactions(:, i) & transactions(:, j)) / size(transactions, 1);
            supp_col1 = sum(transactions(:, i)) / size(transactions, 1);
            supp_col2 = sum(transactions(:, j)) / size(transactions, 1);
            support(i, j) = sup_both;
            confidence(i, j) = sup_both / supp_col1;
            lift(i, j) = sup_both / (supp_col1 * supp_col2);
        end
    end
end

disp("Support for item pairs:");
disp(support);

disp("Confidence for item pairs:");
disp(confidence);

disp("Lift for item pairs:");
disp(lift);

% Generate association rules
disp("Association Rules:");
for i = 1:num_items
    for j = 1:num_items
        if i != j
            if support(i, j) >= min_support
                disp(["Rule: " headers{i} " -> " headers{j}]);
                disp(["  Support: " num2str(support(i, j))]);
                disp(["  Confidence: " num2str(confidence(i, j))]);
                disp(["  Lift: " num2str(lift(i, j))]);
            end
        end
    end
end

