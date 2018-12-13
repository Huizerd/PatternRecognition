% Final Assignment IN4085 Pattern Recognition
% Digit Classification
%
% Group 94: Olle Eriksson, Jesse Hagenaars, Ilmari Vikstrom
% 05-12-2018

% Clear/close everything, set random seed
clear; close all; clc
rng('default')
prwaitbar off

% Add subfolders to path
addpath(genpath(fileparts(which(mfilename))))

%% Load data

% Show some data, this will load digits 0-9 with 25 examples each
sample = prnist(0:9, 1:40:1000);
figure('Name', 'Sample')
show(sample)

% Now load all data
% data = prnist(0:9, 1:1000);

%% Preprocess data

% TODO: do we want to resize twice or only after deslanting?
% TODO: do we want to binarize again after deslanting?

% Make squares
squared = im_box(sample, [5 5 5 5], 1);
figure('Name', 'Square')
show(squared)

% Resize to (50, 50)
resized = im_resize(squared, [50 50]);
figure('Name', 'Correct size')
show(resized)

% Convert to prdataset
preprocessed = prdataset(resized);
dummy = zeros(size(preprocessed));

% Denoise and remove slant
for i = 1:size(resized, 1)
    clean = remove_noise(data2im(resized(i)));
    straight = deslant(clean);
    dummy(i, :) = reshape(im_resize(straight, [50 50]), 1, size(dummy, 2));
end

% Assign to prdataset
preprocessed = setdata(preprocessed, dummy);

figure('Name', 'Clean')
show(preprocessed)


%% HOG feature classification

%img = reshape(preprocessed.data(1,:), 50, 50);
%[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
%[hog_3x3, vis3x3] = extractHOGFeatures(img,'CellSize',[3 3]);
%[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
%[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
%[hog_16x16, vis16x16] = extractHOGFeatures(img,'CellSize',[16 16]);
%[hog_25x25, vis25x25] = extractHOGFeatures(img,'CellSize',[25 25]);


%hogFeatureSize = 36; cellSize = [25 25];
%hogFeatureSize = 144; cellSize = [16 16];
hogFeatureSize = 900; cellSize = [8 8];
%hogFeatureSize = 4356; cellSize = [4 4];
%hogFeatureSize = 8100; cellSize = [3 3];
%hogFeatureSize = 20736; cellSize = [2 2];

[trainingSet, testSet, trainingIndices, testIndices] = gendat(preprocessed, 0.7);

numTrainingImages = size(trainingSet,1);
trainingFeatures = zeros(numTrainingImages, hogFeatureSize, 'single');
for i = 1:numTrainingImages
    trainingFeatures(i, :) = extractHOGFeatures(reshape(trainingSet.data(i,:), 50, 50), 'CellSize', cellSize);  
end
trainingLabels = trainingSet.nlab;
trainingFeatures = double(trainingFeatures);
trainingFeatures = prdataset(trainingFeatures, trainingLabels);

numTestImages = size(testSet,1);
testFeatures = zeros(numTestImages, hogFeatureSize, 'single');
for i = 1:numTestImages
    testFeatures(i, :) = extractHOGFeatures(reshape(testSet.data(i,:), 50, 50), 'CellSize', cellSize);
end
testLabels = testSet.nlab;
testFeatures = double(testFeatures);
testFeatures = prdataset(testFeatures, testLabels);


%% Train a classifier
classifier = bpxnc(trainingFeatures, [100 20], 10000);


%% Evaluate classifier
clfdLabs = testFeatures * classifier * labeld;
cm = confmat(testLabels, clfdLabs)
testc(testFeatures, classifier)


%% Cross-validate
allFeatures = [trainingFeatures; testFeatures];
error = prcrossval(allFeatures, vpc, 5, 1)


%% Visualize the HOG features
figure; 
subplot(2,3,1:3); imshow(img);

subplot(2,3,4);  
plot(vis2x2); 
title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});

subplot(2,3,5);
plot(vis4x4); 
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});

subplot(2,3,6);
plot(vis8x8); 
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});
