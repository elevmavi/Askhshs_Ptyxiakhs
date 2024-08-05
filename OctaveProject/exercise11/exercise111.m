% Define stopwords
stopwords = {'the', 'is', 'in', 'and', 'to', 'a', 'of'};

% Define functions for text processing
function tokens = tokenize_text(text)
  tokens = strsplit(text);
endfunction

function tokens = lowercase_words(tokens)
  tokens = cellfun(@(x) tolower(x), tokens, 'UniformOutput', false);
endfunction

function cleaned_text = remove_special_characters(text)
  cleaned_text = regexprep(text, '[\p{P}\s]', ''); % Remove punctuation and whitespace
endfunction

function words = remove_stopwords(words, stopwords)
  words = setdiff(words, stopwords);
endfunction

function cleaned_text = preprocess_text(text, stopwords)
  tokens = tokenize_text(text);
  tokens = lowercase_words(tokens);
  tokens = remove_stopwords(tokens, stopwords);
  %cleaned_text = strjoin(tokens);
  cleaned_text = remove_special_characters(tokens);
endfunction

% Function to create bag-of-words matrix
function bag_of_words = create_bag_of_words(texts)
  % Extract all unique words
  all_words = strsplit(strjoin(texts));
  unique_words = unique(lower(all_words));
  num_texts = length(texts);
  num_words = length(unique_words);

  bag_of_words = zeros(num_texts, num_words);

  for i = 1:num_texts
    words = strsplit(texts{i});
    for j = 1:length(words)
      word_idx = find(strcmp(unique_words, lower(words{j})));
      if ~isempty(word_idx)
        bag_of_words(i, word_idx) = bag_of_words(i, word_idx) + 1;
      endif
    endfor
  endfor
endfunction

% Function to compute TF-IDF matrix
function tfidf_matrix = compute_tfidf(texts)
  % Create Bag-of-Words
  bag_of_words = create_bag_of_words(texts);
  num_texts = size(bag_of_words, 1);
  num_words = size(bag_of_words, 2);

  % Compute Term Frequencies (TF)
  tf = bag_of_words;

  % Compute Document Frequencies (DF)
  df = sum(bag_of_words > 0, 1);

  % Compute Inverse Document Frequencies (IDF)
  idf = log((num_texts + 1) ./ (df + 1)) + 1;

  % Compute TF-IDF
  tfidf_matrix = bsxfun(@times, tf, idf);
endfunction

% Function to compute cosine similarity
function sim = cosine_similarity(mat1, mat2)
  norm1 = sqrt(sum(mat1 .^ 2, 2));
  norm2 = sqrt(sum(mat2 .^ 2, 2));
  sim = (mat1 * mat2') ./ (norm1 * norm2');
endfunction

% Load and preprocess text files
text1 = fileread('resources/data/text1.txt');
text2 = fileread('resources/data/text2.txt');

cleaned_text1 = preprocess_text(text1, stopwords);
cleaned_text2 = preprocess_text(text2, stopwords);

% Combine texts
texts = {text1, text2};

% Create Bag-of-Words and TF-IDF representations
bag_of_words = create_bag_of_words(texts);
tfidf_matrix = compute_tfidf(texts);

% Display Bag-of-Words
disp('Bag of Words Matrix:');
disp(bag_of_words);

% Compute Cosine Similarity for Bag-of-Words
cosine_sim_bow = cosine_similarity(bag_of_words, bag_of_words);
disp('Cosine Similarity (Bag-of-Words):');
disp(cosine_sim_bow);

% Display TF-IDF Matrix
disp('TF-IDF Matrix:');
disp(tfidf_matrix);

% Compute Cosine Similarity for TF-IDF
cosine_sim_tfidf = cosine_similarity(tfidf_matrix, tfidf_matrix);
disp('Cosine Similarity (TF-IDF):');
disp(cosine_sim_tfidf);

