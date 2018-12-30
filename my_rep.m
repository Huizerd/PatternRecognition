function features = my_rep(raw_data)
% MY_REP Creates 'our' representation, i.e. preprocessing + feature
%   extraction.
%
% Inputs:
% - raw_data: prdatafile containing the raw input data
% - image_size: desired size of output images
%
% Outputs:
% - features: prdataset containing the extracted features
%
% Jesse Hagenaars - 30.12.2018

%% Preprocess data

% Make squares
squared = im_box(raw_data, [5 5 5 5], 1);

% We want to end up with images of 50x50 pixels
image_size = [50 50];

% Denoise and remove slant, then resize
prep_map = squared * filtim('remove_noise') * filtim('deslant') * im_box([5 5 5 5], 1) * im_resize(image_size);

% Put in empty prdataset
prep_map = prdataset(prep_map);
preprocessed = prdataset(prep_map.data);
preprocessed = setfeatsize(preprocessed, image_size);
preprocessed = setname(preprocessed, 'preprocessed NIST');

%% Feature extraction

% Set parameters for HOG
cell_size = [4 4];

% Extract HOG features per image
features = preprocessed * filtim('extractHOGFeatures', {'CellSize', cell_size});

% Adjust HOG prdataset
features = setprior(features, getprior(features)); % avoid warnings
features = setname(features, 'features HOG');

end
