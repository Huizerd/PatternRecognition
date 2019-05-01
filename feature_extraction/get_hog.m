function features = get_hog(preprocessed, cell_size)
% GET_HOG Extracts HOG features.
%
% Inputs:
% - preprocessed: prdataset containing preprocessed data
% - cell_size: size of HOG cells
%
% Outputs:
% - features: prdataset containing the extracted HOG features
%
% . - 02.01.2010

% Extract HOG features per image
features = preprocessed * filtim('extractHOGFeatures', ...
    {'CellSize', cell_size});

% Adjust HOG prdataset
features = setprior(features, getprior(features)); % avoid warnings
features = setname(features, 'features HOG');

end
