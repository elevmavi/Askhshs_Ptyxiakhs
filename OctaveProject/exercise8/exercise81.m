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

function rules = generate_association_rules(frequent_itemsets, min_conf)
    % Generate association rules from frequent itemsets
    rules = {};
    for i = 1:length(frequent_itemsets)
        itemset = frequent_itemsets{i};
        for j = 1:length(itemset)
            antecedent = itemset(1:j);
            consequent = itemset(j+1:end);
            if isempty(consequent)
                continue;
            end

            % Compute support and confidence
            support = length(find(all(data(:, itemset) == 1, 2))) / size(data, 1);
            conf = length(find(all(data(:, itemset) == 1, 2))) / length(find(all(data(:, antecedent) == 1, 2)));

            if conf >= min_conf
                rule = struct();
                rule.antecedent = antecedent;
                rule.consequent = consequent;
                rule.support = support;
                rule.confidence = conf;
                rules{end + 1} = rule;
            end
        end
    end
end

pkg load io;  % For CSV file handling

function process_data(data, min_sup, min_conf, output_filename)
    data = data > 0;  % Convert to binary matrix

    frequent_itemsets = apriori(data, min_sup);
    rules = generate_association_rules(frequent_itemsets, min_conf);

    % Save to file
    fid = fopen(output_filename, 'w');
    fprintf(fid, 'Antecedent\tConsequent\tSupport\tConfidence\n');
    for i = 1:length(rules)
        rule = rules{i};
        antecedent_str = sprintf('%d ', rule.antecedent);
        consequent_str = sprintf('%d ', rule.consequent);
        fprintf(fid, '%s\t%s\t%.2f\t%.2f\n', antecedent_str, consequent_str, rule.support, rule.confidence);
    end
    fclose(fid);
end



bakery1000 = load('resources/data/Bakery1000.mat').xV(:, 2:end);
data = csvread('resources/data/Bakery75000.csv')(:, 2:end);

% Define minimum support and confidence
min_sup = 0.05;
min_conf = 0.1;

% Process data
process_data(bakery1000, min_sup, min_conf, 'BakeryRules-1000-1.txt');
process_data(bakery7500, min_sup, min_conf, 'BakeryRules-7500-1.txt');

min_conf = 0.05;
process_data(bakery1000, min_sup, min_conf, 'BakeryRules-1000-2.txt');
process_data(bakery7500, min_sup, min_conf, 'BakeryRules-7500-2.txt');

min_sup = 0.005;
min_conf = 0.005;
process_data(bakery1000, min_sup, min_conf, 'BakeryRules-1000-3.txt');
process_data(bakery7500, min_sup, min_conf, 'BakeryRules-7500-3.txt');


