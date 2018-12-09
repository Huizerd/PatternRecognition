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
figure
show(sample)

% Now load all data
% data = prnist(0:9, 1:1000);

%% Preprocess data

% Make squares
squared = im_box(sample, [5 5 5 5], 1);

figure
show(squared)

% Remove some noise?

% Do some rectification?
for i = 1:size(squared, 1)
    straight = deslant(data2im(squared(i)));
    imshow(cat(2, im_resize(data2im(squared(i)), [50 50]), im_resize(straight, [50 50])))
    pause
end
