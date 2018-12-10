function clean = remove_noise(dirty)
% REMOVE_NOISE Removes any blobs that have an area less than 50 pixels. 
%   Returns a cleaned image.
%
% Inputs:
% - dirty: image with noise
%
% Outputs:
% - clean: denoised image
%
% Ilmari Vikstrom - 06.12.2018

clean = double(bwareaopen(dirty, 50));

end

