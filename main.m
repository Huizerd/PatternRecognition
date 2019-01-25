% Final Assignment IN4085 Pattern Recognition
% Digit Classification
%
% Group 94: Olle Eriksson, Jesse Hagenaars, Ilmari Vikstrom
% 05-12-2018

% Clear/close everything, set random seed
clear; close all; clc
rng('default')

% Increase max size of datasets (needed to load all objects)
prmemory(100e6)

% Add subfolders to path
addpath(genpath(fileparts(which(mfilename))))

% Add subfolders to path
addpath(genpath(fileparts(which(mfilename))))

%% Load data

% Now load data that should be used for training
% 10:10:1000

%raw_data = gendat(prnist(0:9), 50*ones(10,1)); 
raw_data = prnist(0:9,1:5:1000);

% Show raw data
%figure('Name', 'Data')
%show(raw_data)

%% Preprocess data

% Images of 50x50 pixels, with 5 blank rows/columns
image_size = [50 50];
blanks = [5 5 5 5];

% Do preprocessing
preprocessed = preprocessing(raw_data, image_size, blanks, true);

% Show preprocessed data
figure('Name', 'Preprocessed')
show(preprocessed)

%% Feature extraction

% Get untrained PCA mapping and PCA visualization
components = 100; % first 100 components
[u_pca, pca_vis] = get_pca(preprocessed, components, image_size);

% Show eigendigits
figure('Name', 'Eigendigits')
show(pca_vis)


%% Variables

% Classifiers to be tried out. Apply scaling on HOG, scaling is applied
% through pca-mapping => no scaling of pca-classifiers. 
classifiers_pca = {fisherc,knnc,svc,libsvc};
classifiers_hog = {fisherc, scalem('variance') * knnc, ...
    scalem('variance') * svc, scalem('variance') * libsvc}; 

cell_sizes = 6:14; % cell sizes for HOG
components  = 10:20:90; % # pca components... maybe more comps
nf = 5; % number of folds (cross-validation) 

%% Find the best combination of HOG cell size and number of PCA components 
%  based on cross-validation

lowest = [inf inf inf inf]; 
best_cell_sizes = {};
best_comps = [0 0 0 0];

errors = cell(4,1);

rows = length(components) + 1;
cols = size(cell_sizes, 2) + 1;
errors{1} = zeros(rows, cols); % fisherc
errors{2} = zeros(rows, cols); % knnc
errors{3} = zeros(rows, cols); % svc
errors{4} = zeros(rows, cols); % libsvc

% Fill error matrices column by column 
for i = 1:size(cell_sizes,2) 
    cell_size = cell_sizes(i); 
    features_hog = get_hog(preprocessed, [cell_size, cell_size]);
    
    for c = 1:length(components)
        comps = components(c);
        
        % Do PCA on preprocessed, i.e no HOG, fill eastest column. 
        % This runs only once for every component
        if i == size(cell_sizes, 2) 
            [u_map, ~] = get_pca(preprocessed, comps, image_size);
            error_pca = prcrossval(preprocessed,...
                u_map * classifiers_pca, nf, 1);
            
            errors{1}(c, i+1) = error_pca(1); % fisherc
            errors{2}(c, i+1) = error_pca(2); % knnc
            errors{3}(c, i+1) = error_pca(3); % svc
            errors{4}(c, i+1) = error_pca(4); % libsvc
            
            % Update best results
            [lowest, best_cell_sizes, best_comps] = updateResults(...
                error_pca, comps, 0, lowest, best_cell_sizes, best_comps); 
        end
        
        [u_map, ~] = get_pca(features_hog, comps, image_size);
        error_hog_pca = prcrossval(features_hog,...
            u_map * classifiers_pca, nf, 1);
        
        errors{1}(c, i) = error_hog_pca(1); % fisherc
        errors{2}(c, i) = error_hog_pca(2); % knnc
        errors{3}(c, i) = error_hog_pca(3); % svc
        errors{4}(c, i) = error_hog_pca(4); % libsvc
        
        % update best results
        [lowest, best_cell_sizes, best_comps] = updateResults(...
                error_hog_pca, comps, cell_size, lowest,...
                best_cell_sizes, best_comps); 
    end
    
    % Just use HOG-features, i.e no PCA, fill southest row.
    % This runs once for every cell-size
    error_hog = prcrossval(features_hog, classifiers_hog, nf, 1);
    
    errors{1}(c+1, i) = error_hog(1); % fisherc 
    errors{2}(c+1, i) = error_hog(2); % knnc   
    errors{3}(c+1, i) = error_hog(3); % svc
    errors{4}(c+1, i) = error_hog(4); % libsvc
    
    % update best results
    [lowest, best_cell_sizes, best_comps] = updateResults(...
                error_hog, 0, cell_size, lowest, best_cell_sizes,...
                best_comps);  
end

%% Show errors

% Append labels to error-matrices
for i = 1:length(errors)
    errorMatrix = errors{i};
    errorMatrix = [[cell_sizes 0]; errorMatrix];
    errorMatrix = [[nan, components, 0]' ,errorMatrix];
    errorMatrix(size(errorMatrix,1), size(errorMatrix,1)) = nan;
    errors{i} = errorMatrix;
end

errors_fisherc = errors{1};
errors_knnc = errors{2};
errors_svc = errors{3};
errors_libsvc = errors{4};

% show error matrices
disp('Errors: fisherc')
disp('row: components, col: cell size')
disp(errors_fisherc)
disp('Errors: knnc')
disp('row: components, col: cell size')
disp(errors_knnc)
disp('Errors: svc')
disp('row: components, col: cell size')
disp(errors_svc);
disp('Errors: libsvc')
disp('row: components, col: cell size')
disp(errors_libsvc);


%% Show best settings

disp('best comps')
disp(['fisherc',' ', 'knnc',' ', 'svc', ' ' ,'libsvc'])
disp(best_comps)
disp('best cell sizes:')
disp(['fisherc',' ', 'knnc',' ', 'svc', ' ' ,'libsvc'])
disp(best_cell_sizes)
disp('best errors')
disp(['fisherc',' ', 'knnc',' ', 'svc', ' ' ,'libsvc'])
disp(lowest)

% TESTING!
%% Get features

hog_features = get_hog(preprocessed,[best_cell_size,...
    best_cell_size]);
%% Best settings

global best_cell_size;

best_cell_size = 8; % for all
best_comp = 0; % for svc

u_map = scalem('variance');%*pcam(best_comp);
w = hog_features * (u_map*svc); % get classifier 

%% Benchmark

test_sizes = 100
result = [];
for i = 1:length(test_sizes)
    bench_error = nist_eval('combined_rep', w, test_sizes(i));
    result = [result bench_error];
end
result;

%% Learning curves

best_comp = 60; % for libsvc
u_map = scalem('variance')*pcam(best_comp);
r1 = cleval(hog_features, u_map*libsvc, [10 50 100 200] , 5);

best_comp = 80; % for libsvc
u_map = scalem('variance')*pcam(best_comp);
r2 = cleval(hog_features, u_map*libsvc, [10 50 100 200] , 5);

u_map = scalem('variance'); % 0 components
r3 = cleval(hog_features, u_map*libsvc, [10 50 100 200] , 5);


%r2 = clevalf(hog_features_small, u_map_small*libsvc, 10:10:50,1);
%figure
%plote(r2)
%bench_error_hog_small = nist_eval('hog_pca_rep', w_small, 100);
%disp(bench_error_hog_small)

%% Benchmark

% Train SVM
classifier_hog = libsvc(features_hog);
classifier_pca = preprocessed * (u_pca * libsvc);
classifier_hog_pca = features_hog_large * (u_hog_pca * libsvc);

bench_error_hog = nist_eval('hog_rep', classifier_hog, 100);
disp(bench_error_hog)

bench_error_pca = nist_eval('pca_rep', classifier_pca, 100);
disp(bench_error_pca)

bench_error_hog_pca = nist_eval('combined_rep', classifier_hog_pca, 100);
disp(bench_error_hog_pca)