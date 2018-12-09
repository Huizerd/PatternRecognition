function deslanted = deslant2(slanted)
% DESLANT Deslants/deskews an image.
%
% Inputs:
% - slanted: slanted image, matrix of doubles
%
% Outputs:
% - deslanted: deslanted image, matrix of doubles
%
% Jesse Hagenaars - 06.12.2018

% Get central moments
moments = im_moments(slanted, 'central');

% properties of im_moments
variance_x = moments(1);
variance_y = moments(3);
covariance_xy = moments(2);

if abs(variance_y) < 1e-2
    deslanted = slanted;
else
    skew = covariance_xy / variance_y;
    M = [1 skew -0.5*size(slanted, 1)*skew; 0 1 0; 0 0 1]';
    deslanted = imwarp(slanted, affine2d(M));
end

end