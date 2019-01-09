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

% Load digits 0-9 with 25 examples each
% sample = prnist(0:9, 1:40:1000);
% figure('Name', 'Sample')
% show(sample)

% Now load all data
raw_data = prnist(0:9, 1:2:1000);

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

%% Classify

% TODO: test libsvc
% TODO: test different cell sizes
% TODO: test different nr of PCA components (can't be done with clevalf
%   since PCA is in classifier

% Test various classifiers
classifiers = {fisherc, knnc, svc};
[error_pca, ~, out_pca] = prcrossval(preprocessed, u_pca * classifiers, ...
    5, 1);
[error_hog, ~, out_hog] = prcrossval(features_hog, classifiers, 5, 1);

disp(error_pca)
disp(error_hog)

% Do classifier evaluation for various training set sizes
train_sizes = [5 10 200 300];
error_train_size_pca = cleval(preprocessed, u_pca * classifiers, ...
    train_sizes, 5);
error_train_size_hog = cleval(features_hog, classifiers, train_sizes, 5);

figure('Name', 'Error train size PCA')
plote(error_train_size_pca)
figure('Name', 'Error train size HOG')
plote(error_train_size_hog)

% Do classifier evaluation for various feature set sizes
% feat_size_pca = [1 10 50 100];
% train_test_split = 0.7;
% error_feat_size_pca = clevalf(preprocessed, u_pca * classifiers, ...
%     feat_size_pca, train_test_split, 5);
% 
% figure('Name', 'Error feat size PCA')
% plote(error_feat_size_pca)

%% Benchmark

% Train SVMs
classifier_hog = svc(features_hog);
classifier_pca = preprocessed * (u_pca * svc);

bench_error_hog = nist_eval('hog_rep', classifier_hog, 100);
disp(bench_error_hog)

bench_error_pca = nist_eval('pca_rep', classifier_pca, 100);
disp(bench_error_pca)

% Get datasets

% Scenario 1

% Scenario 2
