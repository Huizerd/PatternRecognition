% Final Assignment IN4085 Pattern Recognition
% Digit Classification
%
% Group 94: Olle Eriksson, Jesse Hagenaars, Ilmari Vikstrom
% 05-12-2018

% Clear/close everything, set random seed
clear; close all; clc
rng('default')
% prwaitbar off

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
raw_data = prnist(0:9, 1:5:1000);

%% Preprocess data

% TODO: do we want to binarize again after deslanting?

% Make squares
squared = im_box(raw_data, [5 5 5 5], 1);
% figure('Name', 'Square')
% show(squared)

% We want to end up with images of 50x50 pixels
image_size = [50 50];
data = zeros(size(squared, 1), prod(image_size));

% Denoise and remove slant, then resize
for i = 1:size(squared, 1)
    clean = remove_noise(data2im(squared(i)));
    straight = deslant(clean);
    data(i, :) = reshape(im_resize(straight, image_size), 1, prod(image_size));
end

% Create prdataset
preprocessed = prdataset(data, squared.labels);
preprocessed = setfeatsize(preprocessed, image_size);
preprocessed = setname(preprocessed, 'preprocessed NIST');

% figure('Name', 'Clean')
% show(preprocessed)

%% Feature extraction

% TODO: PCA mapping only from training data or also validation data?
% TODO: try PCA & Fisher together

% Do PCA
eigendigits = 50; % first 50 eigendigits
pca = pcam(preprocessed, eigendigits);
% figure
% show(pca)
% 
% % Do Fisher
% fish = fisherm(preprocessed);
% figure
% show(fish)
% 
% % Do Karhunen-Loeve
% kh = klm(preprocessed, 50);
% figure
% show(kh)
% 
% % Neither of the above seems to work well!

% Create PCA prdataset
features_pca = setdata(preprocessed, preprocessed * pca);
features_pca = setfeatsize(features_pca, [1 eigendigits]);
features_pca = setprior(features_pca, getprior(features_pca)); % avoid warnings
features_pca = setname(features_pca, 'features PCA');

% TODO: incorporate feature extraction into loop --> my_rep

% Set parameters and create dummy matrix for HOG
cell_size = [4 4];
hog_size = size(extractHOGFeatures(reshape(preprocessed.data(1, :), preprocessed.featsize), 'CellSize', cell_size), 2);
data = zeros(size(preprocessed, 1), hog_size);

% Extract HOG features per image
for i = 1:size(preprocessed, 1)
    data(i, :) = extractHOGFeatures(reshape(preprocessed.data(i, :), preprocessed.featsize), 'CellSize', cell_size);
end

% Create HOG prdataset
features_hog = setdata(preprocessed, data);
features_hog = setfeatsize(features_hog, [1 hog_size]);
features_hog = setprior(features_hog, getprior(features_hog)); % avoid warnings
features_hog = setname(features_hog, 'features HOG');

%% Classify

rng('default')

% TODO: clevalf for various features for HOG and PCA as well? Is this even
%   appropriate for HOG?

% Does one multiply by pca? Or what?
% clfs = {fisherc, knnc, svc};
% [error, std] = prcrossval(preprocessed * pca, clfs, 10, 10);
% disp(error)
% disp(std)

% [error, ~, nlab_out] = prcrossval(preprocessed, fisherc, 10);
% disp(error)
% confmat(getnlab(preprocessed), nlab_out)

% Test various classifiers
features = {features_pca, features_hog};
classifiers = {svc, knnc, loglc};
[error, ~, labels_out] = prcrossval(features, classifiers, 5, 1);
disp(error)

% Do classifier evaluation for various training set sizes
% train_sizes = [10 50 100 200 500];
% error_train_size = cleval(features, classifiers, train_sizes, 10);
% figure
% plote(error_train_size)

%%

% Do classifier evaluation for various feature set sizes
feat_size_pca = [1 5 10 20 50];
train_test_split = 0.7;
error_feat_size_pca = clevalf(features_pca, classifiers, feat_size_pca, train_test_split, 10);
figure
plote(error_feat_size_pca)
