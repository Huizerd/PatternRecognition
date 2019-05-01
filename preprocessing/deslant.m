function deslanted = deslant(slanted)
% DESLANT Deslants/deskews an image. Based on OpenCV implementation:
%   https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_svm/py_svm_
%   opencv/py_svm_opencv.html
%
% Inputs:
% - slanted: slanted image, matrix of doubles
%
% Outputs:
% - deslanted: deslanted image, matrix of doubles
%
% . - 09.12.2018

% Get central image moments
moments = im_moments(slanted, 'central');

var_x = moments(1); % mu_20
var_y = moments(3); % mu_02
cov_xy = moments(2); % mu_11

% Compute horizontal shear factor
alpha = cov_xy / var_y;

% Do transform
tform = affine2d([1 0 0; -alpha 1 0; 0 0 1]);
deslanted = imwarp(slanted, tform);

end