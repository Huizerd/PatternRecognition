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

% Load digits 0-9 with 500 examples each
raw_data = prnist(0:9, 1:2:1000);

% Show data
% figure('Name', 'Data')
% show(raw_data)

%% Preprocess data

% Images of 50x50 pixels, with 5 blank rows/columns
image_size = [50 50];
blanks = [5 5 5 5];

% Do preprocessing
preprocessed = preprocessing(raw_data, image_size, blanks, true);

% Show some
figure('Name', 'Preprocessed')
show(preprocessed)

%% Feature extraction

% Get untrained PCA mapping and PCA visualization
components = 100; % first 100 components
[u_pca, pca_vis] = get_pca(preprocessed, components, image_size);

% Show eigendigits
figure('Name', 'Eigendigits')
show(pca_vis)

% Extract HOG features per image
cell_size = [4 4];
features_hog = get_hog(preprocessed, cell_size);

% Combination of HOG and PCA, larger cell size to allow PCA to capture more
%   global features instead of noise
large_cell_size = [8 8];
features_hog_large = get_hog(preprocessed, large_cell_size);
hog_size = [1 size(features_hog_large, 2)];
[u_hog_pca, ~] = get_pca(features_hog_large, components, hog_size);

%% Cross-validation

% Test various classifiers: k-NN and SVC need scaling!
classifiers_pca = {fisherc, knnc, svc, libsvc}; % since PCA already has scaling
classifiers_hog = {fisherc, scalem('variance') * knnc, ...
    scalem('variance') * svc, scalem('variance') * libsvc};

% [error_pca, ~, ~] = prcrossval(preprocessed, u_pca * classifiers_pca, ...
%     5, 1);
% [error_hog, ~, ~] = prcrossval(features_hog, classifiers_hog, 5, 1);
% [error_hog_pca, ~, ~] = prcrossval(features_hog_large, ...
%     u_hog_pca * classifiers_pca, 5, 1);
% 
% disp(error_pca)
% disp(error_hog)
% disp(error_hog_pca)

% Do classifier evaluation for various training set sizes
% train_sizes = [5 10 200 300];
% error_train_size_pca = cleval(preprocessed, u_pca * classifiers, ...
%     train_sizes, 5);
% error_train_size_hog = cleval(features_hog, classifiers, train_sizes, 5);
% 
% figure('Name', 'Error train size PCA')
% plote(error_train_size_pca)
% figure('Name', 'Error train size HOG')
% plote(error_train_size_hog)

% Do classifier evaluation for various feature set sizes
% feat_size_pca = [1 10 50 100];
% train_test_split = 0.7;
% error_feat_size_pca = clevalf(preprocessed, u_pca * classifiers, ...
%     feat_size_pca, train_test_split, 5);
% 
% figure('Name', 'Error feat size PCA')
% plote(error_feat_size_pca)

%% Feature set validation: HOG cell sizes

% cell_sizes = [4:16];
% errors_hog = {};
% for i = cell_sizes
%     errors_hog{1, size(errors_hog, 2)+1} = [i i];
%     features_hog = get_hog(preprocessed, [i i]);
%     [error_hog, ~, ~] = prcrossval(features_hog, classifiers_hog, 5, 1);
%     errors_hog{2, size(errors_hog, 2)} = error_hog;
% end
% 
% values = errors_hog(2,:);
% lowest = [inf inf inf];
% best_cell_size = {};
% 
% for i = 1:length(values)
%     p = values{i};
%     if p(1) < lowest(1)
%         lowest(1) = p(1);
%         best_cell_size{1} = errors_hog{1,i};
%     end
%     if p(2) < lowest(2)
%         lowest(2) = p(2);
%         best_cell_size{2} = errors_hog{1,i};
%     end
%     if p(3) < lowest(3)
%         lowest(3) = p(3);
%         best_cell_size{3} = errors_hog{1,i};
%     end
% end

% Then call
% features_hog = get_hog(preprocessed, cell_size);
% with the best cell_size!!!

%% Training set size

% Experiment with these training set sizes, remaining samples are used for
%   testing!
train_sizes = [5 10 50 100 200];

% Fisher
features_fisher = get_hog(preprocessed, [8 8]);
comp_fisher = [70 90];
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
features_knn = get_hog(preprocessed, [8 8]);
[pca_knn, ~] = get_pca(features_knn, 70, [1 size(features_knn, 2)]);
e_train_knn = cleval(features_knn, pca_knn * knnc, train_sizes, 5);

figure('Name', 'Error train: k-NN')
plote(e_train_knn)
set(gca, 'XScale', 'linear')

% SVC (svc)
features_svc = get_hog(preprocessed, [8 8]);
e_train_svc = cleval(features_svc, scalem('variance') * svc, ...
    train_sizes, 5);

figure('Name', 'Error train: SVC (svc)')
plote(e_train_svc)
set(gca, 'XScale', 'linear')

% SVC (libsvc)
features_libsvc = get_hog(preprocessed, [8 8]);
[pca_libsvc, ~] = get_pca(features_libsvc, 60, ...
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
            e_train_all.(f{1}) = char(vertcat(cellstr(e_train_all.(f{1})), ...
                cellstr(s.(f{1}))));
        end
    end
end

figure('Name', 'Error train: all')
plote(e_train_all)
set(gca, 'XScale', 'linear')

%% Benchmark

% Train SVM
% classifier_hog = libsvc(features_hog);
% classifier_pca = preprocessed * (u_pca * libsvc);
% classifier_hog_pca = features_hog_large * (u_hog_pca * libsvc);

% bench_error_hog = nist_eval('hog_rep', classifier_hog, 100);
% disp(bench_error_hog)
% 
% bench_error_pca = nist_eval('pca_rep', classifier_pca, 100);
% disp(bench_error_pca)
% 
% bench_error_hog_pca = nist_eval('combined_rep', classifier_hog_pca, 100);
% disp(bench_error_hog_pca)

% Get datasets

% Scenario 1

% Scenario 2

%% Live classification

% Train on 200
raw_data = prnist(0:9, 1:5:1000);
preprocessed = preprocessing(raw_data, [50 50], [5 5 5 5], true);
features = get_hog(preprocessed, [8 8]);

% Classifiers
% [pca_fisher, ~] = get_pca(features, 90, [1 size(features, 2)]);
[pca_libsvc, ~] = get_pca(features, 60, [1 size(features, 2)]);
% classifier_fisher = features * (pca_fisher * fisherc);
% classifier_svc = features * (scalem('variance') * svc);
classifier_libsvc = features * (pca_libsvc * libsvc);

% Hard-coded answers to questions the script will ask
% n_digits = [75 77 80 77 83 85 80 89 90 88];

% Get scanned digits
live_data = get_live_digits('data/', [50 50], n_digits);

% Show how ugly my writing is
figure('Name', 'Live digits')
show(live_data)

% Feature extraction
features_live = get_hog(live_data, [8 8]);

% Classify
% error_live = features_live * ...
%     {classifier_fisher, classifier_svc, classifier_libsvc} * testc;
% disp(error_live)

% Get confusion matrix for best
confmat(features_live * classifier_libsvc)
% confmat(features_live * classifier_svc)

% Do classifier evaluation for various training set sizes
% train_sizes = [5 10 50 100 200];
% untrained_clf = {pca_fisher * fisherc, scalem('variance') * svc, ...
%     pca_libsvc * libsvc};
% e_train_size_live = cleval(features, untrained_clf, train_sizes, 5, ...
%     features_live);
% 
% figure('Name', 'Error train size live digits')
% plote(e_train_size_live)
% set(gca, 'XScale', 'linear')




