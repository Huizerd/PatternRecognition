% Final Assignment IN4085 Pattern Recognition
% Digit Classification
%
% Group 94: Olle Eriksson, Jesse Hagenaars, Ilmari Vikstrom
% 05-12-2018

% Clear/close everything, set random seed
clear; close all; clc
rng('default')

% Increase max size of datasets (needed to load all objects)
prmemory(100e6)

% Add subfolders to path
addpath(genpath(fileparts(which(mfilename))))

%% Load data

% Load raw datasets of 500 samples per digit class
raw_data = prnist(0:9, 1:2:1000);

% Show raw data
figure('Name', 'Data')
show(raw_data)

%% Preprocess data

% Images of 50x50 pixels, with 5 blank rows/columns
image_size = [50 50];
blanks = [5 5 5 5];

% Do preprocessing
preprocessed = preprocessing(raw_data, image_size, blanks, true);

% Show preprocessed data
figure('Name', 'Preprocessed')
show(preprocessed)

%% Cross-validation for HOG cell size and PCA components 

% Classifiers to be tried out, k-NN and SVC need scaling, which is included
%   in PCA but not in HOG!
classifiers_pca = {fisherc, knnc, svc, libsvc};
classifiers_hog = {fisherc, scalem('variance') * knnc, ...
    scalem('variance') * svc, scalem('variance') * libsvc}; 

% Configuration
cell_sizes = 6:14; % cell sizes for HOG
components  = 10:10:90; % pca components... maybe more comps
nf = 5; % number of folds (cross-validation)
[preprocessed_50, ~, ~, ~] = gendat(preprocessed, ones(1, 10) * 50);

% Trackers
lowest = [inf inf inf inf]; 
best_cell_sizes = {};
best_comps = [0 0 0 0];

% Building error matrices
errors = cell(4, 1);
rows = length(components) + 1;
cols = size(cell_sizes, 2) + 1;
errors{1} = zeros(rows, cols); % fisherc
errors{2} = zeros(rows, cols); % knnc
errors{3} = zeros(rows, cols); % svc
errors{4} = zeros(rows, cols); % libsvc

% Fill error matrices column by column 
for i = 1:size(cell_sizes, 2)
    cell_size = cell_sizes(i);
    features_hog = get_hog(preprocessed_50, [cell_size cell_size]);
    
    for c = 1:length(components)
        comps = components(c);
        
        if i == size(cell_sizes, 2)
            % Do PCA on preprocessed, i.e no HOG, fill rightmost column
            [u_map, ~] = get_pca(preprocessed_50, comps, image_size);
            error_pca = prcrossval(preprocessed_50, ...
                u_map * classifiers_pca, nf, 5);
            
            errors{1}(c, i+1) = error_pca(1); % fisherc
            errors{2}(c, i+1) = error_pca(2); % knnc
            errors{3}(c, i+1) = error_pca(3); % svc
            errors{4}(c, i+1) = error_pca(4); % libsvc
            
            % Update best results
            [lowest, best_cell_sizes, best_comps] = update_results(...
                error_pca, comps, 0, lowest, best_cell_sizes, best_comps);
            
        else
            % Do PCA on HOG
            [u_map, ~] = get_pca(features_hog, comps, image_size);
            error_hog_pca = prcrossval(features_hog, ...
                u_map * classifiers_pca, nf, 5);

            errors{1}(c, i) = error_hog_pca(1); % fisherc
            errors{2}(c, i) = error_hog_pca(2); % knnc
            errors{3}(c, i) = error_hog_pca(3); % svc
            errors{4}(c, i) = error_hog_pca(4); % libsvc
            
            % Update best results
            [lowest, best_cell_sizes, best_comps] = update_results(...
                error_hog_pca, comps, cell_size, lowest, ...
                best_cell_sizes, best_comps);
        end
    end
    
    % Just use HOG, i.e no PCA, fill bottom row
    error_hog = prcrossval(features_hog, classifiers_hog, nf, 5);
    
    errors{1}(c+1, i) = error_hog(1); % fisherc 
    errors{2}(c+1, i) = error_hog(2); % knnc   
    errors{3}(c+1, i) = error_hog(3); % svc
    errors{4}(c+1, i) = error_hog(4); % libsvc
    
    % update best results
    [lowest, best_cell_sizes, best_comps] = update_results(...
        error_hog, 0, cell_size, lowest, best_cell_sizes, best_comps);  
end

%% Show results

% Append labels to error-matrices
for i = 1:length(errors)
    error_matrix = errors{i};
    error_matrix = [[cell_sizes 0]; error_matrix];
    error_matrix = [[nan, components, 0]' ,error_matrix];
    error_matrix(size(error_matrix,1), size(error_matrix,1)) = nan;
    errors{i} = error_matrix;
end

errors_fisherc = errors{1};
errors_knnc = errors{2};
errors_svc = errors{3};
errors_libsvc = errors{4};

% Show error matrices
disp('Errors: fisherc')
disp('row: components, col: cell size')
disp(errors_fisherc)
disp('Errors: knnc')
disp('row: components, col: cell size')
disp(errors_knnc)
disp('Errors: svc')
disp('row: components, col: cell size')
disp(errors_svc);
disp('Errors: libsvc')
disp('row: components, col: cell size')
disp(errors_libsvc);

% Show best settings
disp('best comps')
disp(['fisherc',' ', 'knnc',' ', 'svc', ' ' ,'libsvc'])
disp(best_comps)
disp('best cell sizes:')
disp(['fisherc',' ', 'knnc',' ', 'svc', ' ' ,'libsvc'])
disp(best_cell_sizes)
disp('best errors')
disp(['fisherc',' ', 'knnc',' ', 'svc', ' ' ,'libsvc'])
disp(lowest)

%% Select best per type of classifier

global best_cell_size
global best_comp

best_cell_size = [8 8 8 8];
best_comp = {[70 90], 70, 0, 60};

%% Learning curves

% Experiment with these training set sizes, remaining samples are used for
%   testing!
train_sizes = [5 10 50 100 200];
[preprocessed_500, ~, ~, ~] = gendat(preprocessed, ones(1, 10) * 500);

% Fisher
features_fisher = get_hog(preprocessed_500, ...
    [best_cell_size(1) best_cell_size(1)]);
comp_fisher = best_comp{1};
for i = 1:length(comp_fisher)
    if comp_fisher(i) ~= 0
        [clf_fisher{1, i}, ~] = get_pca(features_fisher, ...
            comp_fisher(i), [1 size(features_fisher, 2)]);
    else
        clf_fisher{1, i} = 1;
    end
end

e_train_fisher = cleval(features_fisher, clf_fisher * fisherc, ...
    train_sizes, 5);

figure('Name', 'Error train: Fisher')
plote(e_train_fisher)
set(gca, 'XScale', 'linear')

% k-NN
features_knn = get_hog(preprocessed, ...
    [best_cell_size(2) best_cell_size(2)]);
[pca_knn, ~] = get_pca(features_knn, best_comp{2}, ...
    [1 size(features_knn, 2)]);
e_train_knn = cleval(features_knn, pca_knn * knnc, train_sizes, 5);

figure('Name', 'Error train: k-NN')
plote(e_train_knn)
set(gca, 'XScale', 'linear')

% SVC (svc)
features_svc = get_hog(preprocessed, ...
    [best_cell_size(3) best_cell_size(3)]);
e_train_svc = cleval(features_svc, scalem('variance') * svc, ...
    train_sizes, 5);

figure('Name', 'Error train: SVC (svc)')
plote(e_train_svc)
set(gca, 'XScale', 'linear')

% SVC (libsvc)
features_libsvc = get_hog(preprocessed, ...
    [best_cell_size(4) best_cell_size(4)]);
[pca_libsvc, ~] = get_pca(features_libsvc, best_comp{4}, ...
    [1 size(features_libsvc, 2)]);
e_train_libsvc = cleval(features_libsvc, pca_libsvc * libsvc, ...
    train_sizes, 5);

figure('Name', 'Error train: SVC (libsvc)')
plote(e_train_libsvc)
set(gca, 'XScale', 'linear')

% Create master plot
e_train_all = e_train_fisher;
for s = [e_train_knn e_train_svc e_train_libsvc]
    for f = {'error', 'std', 'apperror', 'appstd', 'names'}
        
        if ~strcmp(f{1}, 'names')
            e_train_all.(f{1}) = vertcat(e_train_all.(f{1}), s.(f{1}));
        else
            e_train_all.(f{1}) = char(vertcat(cellstr(...
                e_train_all.(f{1})), cellstr(s.(f{1}))));
        end
    end
end

figure('Name', 'Error train: all')
plote(e_train_all)
set(gca, 'XScale', 'linear')

%% Slim down to best 3 classifiers for small and large training sets

best_cell_size = 8; % same for all
best_comp = [90 0 60]; % fisherc, svc, libsvc

%% Benchmark

% Train classifiers for both scenarios
bench_sizes = [10 200];
bench_clfs_base = {fisherc, svc, libsvc};
bench_clfs_untrained = {};
bench_clfs_trained = {};

for i = 1:length(bench_sizes)
    [preprocessed_bench, ~, ~, ~] = gendat(preprocessed, ...
        ones(1, 10) * bench_sizes(i));
    features_bench = get_hog(preprocessed_bench, ...
        [best_cell_size best_cell_size]);
    for j = 1:length(bench_clfs_base)
        [pca_bench, ~] = get_pca(features_bench, best_comp(j), ...
            [1 size(features_bench, 2)]);
        bench_clfs_untrained{1, i*j} = pca_bench * bench_clfs_base{j};
        bench_clfs_trained{1, i*j} = features_bench * ...
            (pca_bench * bench_clfs_base{j});
    end
end
        
% Get benchmark error
for w = bench_clfs_trained
    bench_error = nist_eval('combined_rep', w{1}, 100);
    disp(bench_error)
end

%% Live classification

% Train on 200
[preprocessed_200, ~, ~, ~] = gendat(preprocessed, ones(1, 10) * 200);
features_200 = get_hog(preprocessed_200, [best_cell_size best_cell_size]);
live_clfs_trained = features_200 * bench_clfs_untrained(4:6);

% Hard-coded answers to questions the script will ask
n_digits = [75 77 80 77 83 85 80 89 90 88];

% Get scanned digits
live_data = get_live_digits('data/', image_size, n_digits);

% Show how ugly my writing is
figure('Name', 'Live digits')
show(live_data)

% Feature extraction
features_live = get_hog(live_data, [best_cell_size best_cell_size]);

% Classify
error_live = features_live * live_clfs_trained * testc;
disp(error_live)

% Get confusion matrix for libsvc
confmat(features_live * live_clfs_trained{3});

% Do classifier evaluation for various training set sizes
train_sizes = [5 10 50 100 200];
[preprocessed_250, ~, ~, ~] = gendat(preprocessed, ones(1, 10) * 250);
features_250 = get_hog(preprocessed_250, [best_cell_size best_cell_size]);
e_train_size_live = cleval(features_250, bench_clfs_untrained(4:6), train_sizes, 5, ...
    features_live);

figure('Name', 'Error train size live digits')
plote(e_train_size_live)
set(gca, 'XScale', 'linear')
yticks(gca, 0:0.1:1)
