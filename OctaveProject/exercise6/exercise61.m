pkg load statistics; % Required for statistical functions

function [mean_val, std_val] = calculate_mean_std(attribute, gender, genders)
  % Calculate mean and standard deviation for a given attribute and gender

  values = attribute(strcmp(genders, gender));
  mean_val = mean(values);
  std_val = std(values);
endfunction

function probability = calculate_probability(mean_val, std_val, x)
  % Calculate the Gaussian probability density function for a given value

  probability = (1 / (std_val * sqrt(2 * pi))) * exp(-0.5 * ((x - mean_val) / std_val)^2);
endfunction

% Training dataset
data = struct();
data.gender = {'M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'};
data.height = [182, 180, 170, 180, 152, 167, 165, 175];
data.weight = [81, 86, 77, 74, 45, 68, 58, 68];
data.shoe_size = [45, 42, 45, 40, 30, 35, 32, 37];

% Convert data to matrices
genders = data.gender;
heights = data.height;
weights = data.weight;
shoe_sizes = data.shoe_size;

% Calculate mean and standard deviation for each attribute and gender
[male_height_mean, male_height_std] = calculate_mean_std(heights, 'M', genders);
[male_weight_mean, male_weight_std] = calculate_mean_std(weights, 'M', genders);
[male_shoe_size_mean, male_shoe_size_std] = calculate_mean_std(shoe_sizes, 'M', genders');

[female_height_mean, female_height_std] = calculate_mean_std(heights, 'F', genders);
[female_weight_mean, female_weight_std] = calculate_mean_std(weights, 'F', genders);
[female_shoe_size_mean, female_shoe_size_std] = calculate_mean_std(shoe_sizes, 'F', genders);

% Print results
printf("Males - Height: mean=%f, std=%f\n", male_height_mean, male_height_std);
printf("Males - Weight: mean=%f, std=%f\n", male_weight_mean, male_weight_std);
printf("Males - Shoe Size: mean=%f, std=%f\n", male_shoe_size_mean, male_shoe_size_std);

printf("Females - Height: mean=%f, std=%f\n", female_height_mean, female_height_std);
printf("Females - Weight: mean=%f, std=%f\n", female_weight_mean, female_weight_std);
printf("Females - Shoe Size: mean=%f, std=%f\n", female_shoe_size_mean, female_shoe_size_std);

% Test data
test_data = struct();
test_data.height = 182;
test_data.weight = 58;
test_data.shoe_size = 35;

% Calculate probabilities for males
p_male_height = calculate_probability(male_height_mean, male_height_std, test_data.height);
p_male_weight = calculate_probability(male_weight_mean, male_weight_std, test_data.weight);
p_male_shoe_size = calculate_probability(male_shoe_size_mean, male_shoe_size_std, test_data.shoe_size);

% Total probability for males
p_male = p_male_height * p_male_weight * p_male_shoe_size * 0.5;

% Calculate probabilities for females
p_female_height = calculate_probability(female_height_mean, female_height_std, test_data.height);
p_female_weight = calculate_probability(female_weight_mean, female_weight_std, test_data.weight);
p_female_shoe_size = calculate_probability(female_shoe_size_mean, female_shoe_size_std, test_data.shoe_size);

% Total probability for females
p_female = p_female_height * p_female_weight * p_female_shoe_size * 0.5;

% Print results
printf("Probability of being male: %.20f\n", p_male);
printf("Probability of being female: %.20f\n", p_female);

% Classification
if (p_male > p_female)
  printf("The person is more likely to be male.\n");
else
  printf("The person is more likely to be female.\n");
endif

