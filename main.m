% Final Assignment IN4085 Pattern Recognition
% Digit Classification
%
% Group 94: Olle Eriksson, Jesse Hagenaars, Ilmari Vikstrom
% 05-12-2018

% Clear/close everything, set random seed
clear; close all; clc
rng('default')
prwaitbar off

%% Load data

% Show some data, this will load digits 0-9 with 25 examples each
sample = prnist(0:9, 1:40:1000);
figure
show(sample)

% Now load all data
% data = prnist(0:9, 1:1000);

%% Preprocess data

% Make squares
squared = im_box(sample, [5 5 5 5], 1);

figure
show(squared)

%% Resize
resized = im_resize(squared, [50 50]);

figure
show(resized)

%% Remove noise from the samples. Warning: takes a while.
shape = size(resized)
cleanedImages = {}
for i = 1:shape(1)
    % Convert individual image from prdatafile to a 50x50 logical matrix.
    prdatafileAsMatrix = logical(data2im(resized(i,:)))
    % Remove noise from the 50x50 matrix.
    cleanedImage = removeNoise(prdatafileAsMatrix);
    % Insert cleaned image into cell array.
    cleanedImages{i} = cleanedImage;
end