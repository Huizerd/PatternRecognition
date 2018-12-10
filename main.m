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
sample = prnist(0:9, 1:100:1000);
figure('Name', 'Sample')
show(sample)

% Now load all data
% data = prnist(0:9, 1:1000);

%% Preprocess data

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
%     dummy(1, :) = reshape(im_resize(straight, [50 50]), 1, size(dummy, 2));
    dummy(1, :) = reshape(clean, 1, size(dummy, 2));
end

% Assign to prdataset
setdata(preprocessed, dummy)

figure('Name', 'Clean')
show(preprocessed)
