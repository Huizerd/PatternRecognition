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

% Load digits 0-9 with 25 examples each
% sample = prnist(0:9, 1:40:1000);
% figure('Name', 'Sample')
% show(sample)

% Now load all data
raw_data = gendat(prnist(0:9), 50*ones(10,1)); 
%raw_data = prnist(0:9, 1:100:1000);
%raw_data = prnist(0:9, 1:100:1000);

% Show data
% figure('Name', 'Data')
% show(raw_data)

%% Preprocess data

% Images of 50x50 pixels, with 5 blank rows/columns
image_size = [50 50];
blanks = [5 5 5 5];

% Do preprocessing
preprocessed = preprocessing(raw_data, image_size, blanks, true);

% Show some
figure('Name', 'Preprocessed')
show(preprocessed)

%% Feature extraction

% Get untrained PCA mapping and PCA visualization
components = 100; % first 100 components
[u_pca, pca_vis] = get_pca(preprocessed, components, image_size);

% Show eigendigits
figure('Name', 'Eigendigits')
show(pca_vis)

% Extract HOG features per image
cell_size = [4 4];
features_hog = get_hog(preprocessed, cell_size);

%% Variables

classifiers_pca = {fisherc,knnc,svc,libsvc};
classifiers_hog = {fisherc, scalem('variance') * knnc, ...
    scalem('variance') * svc, scalem('variance') * libsvc}; 
% should we really do scaling? better results?

cell_sizes = 6:14; % cell sizes for HOG
components  = [10:10:90]; % # pca components... maybe more comps
nf = 5; % # folds 

%% Try classifiers on just preprocessed

error = prcrossval(preprocessed, scalem('variance')*classifiers_pca, nf, 1);

%% Find the best combination of HOG cell size and number of PCA components 
%  based on cross-validation

lowest = [inf inf inf inf]; 
best_cell_sizes = {};
best_comps = [0 0 0 0];

errors = cell(4,1);
errors{1} = zeros(length(components)+1, size(cell_sizes,2)+1); % fisherc
errors{2} = zeros(length(components)+1, size(cell_sizes,2)+1); % knnc
errors{3} = zeros(length(components)+1, size(cell_sizes,2)+1); % svc
errors{4} = zeros(length(components)+1, size(cell_sizes,2)+1); % libsvc

% fill matrix column by column 

% now based on small dataset
for i = 1:size(cell_sizes,2) 
    disp('i=')
    disp(i)
    cell_size = cell_sizes(i); 
    features_hog = get_hog(preprocessed, [cell_size, cell_size]);
    for c = 1:length(components)
        comps = components(c);
        
        % do pca on preprocessed, fill eastest column
        if i == size(cell_sizes, 2) 
            [u_map, ~] = get_pca(preprocessed, comps, image_size);
            error_pca = prcrossval(preprocessed,...
                u_map * classifiers_pca, nf, 1);
            errors{1}(c, i+1) = error_pca(1);
            errors{2}(c, i+1) = error_pca(2);
            errors{3}(c, i+1) = error_pca(3);
            errors{4}(c, i+1) = error_pca(4);
            
            % update results
            [lowest, best_cell_sizes, best_comps] = updateResults(...
                error_pca, comps, 0, lowest, best_cell_sizes, best_comps); 
        end
        
        [u_map, ~] = get_pca(features_hog, comps, image_size);
        error_hog_pca = prcrossval(features_hog,...
            u_map * classifiers_pca, nf, 1);
        errors{1}(c, i) = error_hog_pca(1); % fill error-matrices
        errors{2}(c, i) = error_hog_pca(2);
        errors{3}(c, i) = error_hog_pca(3);
        errors{4}(c, i) = error_hog_pca(4);
        [lowest, best_cell_sizes, best_comps] = updateResults(...
                error_hog_pca, comps, cell_size, lowest,...
                best_cell_sizes, best_comps); 
    end
    
    % do only HOG, no pca
    % fill southest row
    error_hog = prcrossval(features_hog, classifiers_hog, nf, 1);
    errors{1}(c+1, i) = error_hog(1); % errors w.o pca, just HOG    
    errors{2}(c+1, i) = error_hog(2);    
    errors{3}(c+1, i) = error_hog(3);
    errors{4}(c+1, i) = error_hog(4);
    
    [lowest, best_cell_sizes, best_comps] = updateResults(...
                error_hog, 0, cell_size, lowest, best_cell_sizes,...
                best_comps);  
end

%% Just use HOG
%  based on cross-validation

lowest = [inf inf inf inf]; 
best_cell_sizes = {};
best_comps = [0 0 0 0];

errors = cell(4,1);
errors{1} = zeros(size(cell_sizes,2),1); % fisherc
errors{2} = zeros(size(cell_sizes,2),1); % knnc
errors{3} = zeros(size(cell_sizes,2),1); % svc
errors{4} = zeros(size(cell_sizes,2),1); % libsvc

% fill matrix column by column 
for i = 1:size(cell_sizes,2) 
    cell_size = cell_sizes(i); 
    features_hog = get_hog(preprocessed, [cell_size, cell_size]);
    
    error_hog = prcrossval(features_hog, classifiers_hog, nf, 1);
    errors{1}(i) = error_hog(1); % errors w.o pca, just HOG    
    errors{2}(i) = error_hog(2);    
    errors{3}(i) = error_hog(3);
    errors{4}(i) = error_hog(4);
    
    [lowest, best_cell_sizes, best_comps] = updateResults(...
                error_hog, 0, cell_size, lowest, best_cell_sizes,...
                best_comps);  
end


for i = 1:length(errors)
    errors{i} = [cell_sizes;(errors{i})'];
    disp(errors{i})
end



%% Show result 

% append labels to error-matrices
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


%%

% show best settings
disp('best comps')
disp(['fisherc',' ', 'knnc',' ', 'svc', ' ' ,'libsvc'])
disp(best_comps)
disp('best cell sizes:')
disp(['fisherc',' ', 'knnc',' ', 'svc', ' ' ,'libsvc'])
disp(best_cell_sizes)
disp('best errors')
disp(['fisherc',' ', 'knnc',' ', 'svc', ' ' ,'libsvc'])
disp(lowest)

%% Variables for testing 

global best_cell_size;
best_cell_size = 8; % for svc

best_comp = 80; % for libsvc
% Right order of attachment of mappings? 

u_map_small = scalem('variance')*pcam(best_comp);
u_map_big = scalem('variance')*pcam(best_comp);

%% Small - benchmark

hog_features_small = get_hog(preprocessed,[best_cell_size,...
    best_cell_size]);

w_small = hog_features_small * (u_map_small*libsvc);

%% plot for smaller data 

r = cleval(hog_features_small, u_map_small*libsvc, 10:10:60,5);
plote(r);

%r2 = clevalf(hog_features_small, u_map_small*libsvc, 10:10:50,1);
%figure
%plote(r2)
%bench_error_hog_small = nist_eval('hog_pca_rep', w_small, 100);
%disp(bench_error_hog_small)

%%

% Combination of HOG and PCA, larger cell size to allow PCA to capture more
%   global features instead of noise
features_hog_large = get_hog(preprocessed, [8 8]);
hog_size = [1 size(features_hog_large, 2)];
[u_hog_pca, ~] = get_pca(features_hog_large, components, hog_size);

%% Cross-validation

% Test various classifiers: k-NN and SVC need scaling!
classifiers_pca = {fisherc, knnc, libsvc}; % since PCA already has scaling
classifiers_hog = {fisherc, scalem('variance') * knnc, ...
    scalem('variance') * libsvc};

[error_pca, ~, ~] = prcrossval(preprocessed, u_pca * classifiers_pca, ...
    5, 1);
[error_hog, ~, ~] = prcrossval(features_hog, classifiers_hog, 5, 1);
[error_hog_pca, ~, ~] = prcrossval(features_hog_large, ...
    u_hog_pca * classifiers_pca, 5, 1);

disp(error_pca)
disp(error_hog)
disp(error_hog_pca)

% Do classifier evaluation for various training set sizes
% train_sizes = [5 10 200 300];
% error_train_size_pca = cleval(preprocessed, u_pca * classifiers, ...
%     train_sizes, 5);
% error_train_size_hog = cleval(features_hog, classifiers, train_sizes, 5);
% 
% figure('Name', 'Error train size PCA')
% plote(error_train_size_pca)
% figure('Name', 'Error train size HOG')
% plote(error_train_size_hog)

% Do classifier evaluation for various feature set sizes
% feat_size_pca = [1 10 50 100];
% train_test_split = 0.7;
% error_feat_size_pca = clevalf(preprocessed, u_pca * classifiers, ...
%     feat_size_pca, train_test_split, 5);
% 
% figure('Name', 'Error feat size PCA')
% plote(error_feat_size_pca)

%% Feature set validation: HOG cell sizes

% cell_sizes = [4:16];
% errors_hog = {};
% for i = cell_sizes
%     errors_hog{1, size(errors_hog, 2)+1} = [i i];
%     features_hog = get_hog(preprocessed, [i i]);
%     [error_hog, ~, ~] = prcrossval(features_hog, classifiers_hog, 5, 1);
%     errors_hog{2, size(errors_hog, 2)} = error_hog;
% end
% 
% values = errors_hog(2,:);
% lowest = [inf inf inf];
% best_cell_size = {};
% 
% for i = 1:length(values)
%     p = values{i};
%     if p(1) < lowest(1)
%         lowest(1) = p(1);
%         best_cell_size{1} = errors_hog{1,i};
%     end
%     if p(2) < lowest(2)
%         lowest(2) = p(2);
%         best_cell_size{2} = errors_hog{1,i};
%     end
%     if p(3) < lowest(3)
%         lowest(3) = p(3);
%         best_cell_size{3} = errors_hog{1,i};
%     end
% end

% Then call
% features_hog = get_hog(preprocessed, cell_size);
% with the best cell_size!!!

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

% Get datasets

% Scenario 1

% Scenario 2





