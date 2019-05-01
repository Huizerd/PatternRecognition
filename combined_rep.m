function features = combined_rep(raw_data)
% HOG_REP Creates a combined representation of HOG and PCA, i.e. 
%   preprocessing + HOG + PCA.
%
% Inputs:
% - raw_data: prdatafile containing the raw input data
%
% Outputs:
% - features: prdataset containing the extracted features
%
% . - 30.12.2018

global best_cell_size

%% Preprocess data

% Images of 50x50 pixels, with 5 blank rows/columns
image_size = [50 50];
blanks = [5 5 5 5];

% Do preprocessing
preprocessed = preprocessing(raw_data, image_size, blanks, false);

%% Feature extraction

% Extract HOG features per image
%cell_size = [8 8];
%features_hog = get_hog(preprocessed, cell_size);
features_hog = get_hog(preprocessed, [best_cell_size best_cell_size]);

% PCA mapping is part of the trained classifier!
features = features_hog;

end
