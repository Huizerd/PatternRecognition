function features = pca_rep(raw_data)
% PCA_REP Creates a PCA representation, i.e. preprocessing + PCA.
%
% Inputs:
% - raw_data: prdatafile containing the raw input data
%
% Outputs:
% - features: prdataset containing the extracted features
%
% . - 30.12.2018

%% Preprocess data

% Images of 50x50 pixels, with 5 blank rows/columns
image_size = [50 50];
blanks = [5 5 5 5];

% Do preprocessing
preprocessed = preprocessing(raw_data, image_size, blanks, false);

%% Feature extraction

% PCA mapping is part of the trained classifier!
features = preprocessed;

end
