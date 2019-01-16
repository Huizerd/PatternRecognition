function features = hog_rep(raw_data)
% HOG_REP Creates a HOG representation, i.e. preprocessing + HOG.
%
% Inputs:
% - raw_data: prdatafile containing the raw input data
%
% Outputs:
% - features: prdataset containing the extracted features
%
% Jesse Hagenaars - 30.12.2018

%% Preprocess data

% Images of 50x50 pixels, with 5 blank rows/columns
image_size = [50 50];
blanks = [5 5 5 5];

% Do preprocessing
preprocessed = preprocessing(raw_data, image_size, blanks, false);

%% Feature extraction

% Extract HOG features per image
cell_size = [4 4];
features = get_hog(preprocessed, cell_size);

end
