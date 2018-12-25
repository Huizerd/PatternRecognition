function live_digits = get_live_digits(scan_dir, image_size)
% GET_LIVE_DIGITS Processes scanned images containing handwritten digits.
%
% Inputs:
% - scan_dir: location containing scanned image(s) of several handwritten
%   digits
% - image_size: desired size of output images
%
% Outputs:
% - live_digits: prdataset of separate binarized digits
%
% Jesse Hagenaars - 25.12.2018

% TODO: check/think about different sizes of digits

% Load scans and collect info
scan_files = dir([scan_dir '*.jpg']);
for i = 1:size(scan_files, 1)
    
    % Get images and labels
    scans{i, 1} = imread([scan_dir scan_files(i).name]);
    label = strsplit(scan_files(i).name, '.'); % take off .jpg
    
    % Show for clarity
    imshow(scans{i})
    title(label{1})
    
    % Get info about what's in the image
    scans{i, 2} = input('Does the image contain one digit class, and is the label correct (e.g. digit_0.jpg)? [yes/no] ', 's');
    scans{i, 3} = label{1};
    scans{i, 4} = input('Number of objects in this image: ');
end

% Close all image windows
close all

% Now process
for i = 1:size(scans, 1)
    if strcmpi('yes', scans{i, 2})
        
        % Binarize image, retain the N largest components
        binarized = bwareafilt(imcomplement(imbinarize(scans{i, 1}, 'global')), scans{i, 4});
        
        % Separate these components and give them labels
        digits = labelmatrix(bwconncomp(binarized));
        
        % Go over components and do preprocessing
        data = zeros(scans{i, 4}, prod(image_size));
        for j = 1:scans{i, 4}
            
            % Make square
            square = im_box(double(digits == j), [5 5 5 5], 1);
            
            % Denoise and remove slant, then resize
            clean = remove_noise(square);
            straight = deslant(clean);
            data(j, :) = reshape(im_resize(straight, image_size), 1, prod(image_size));
        end
        
        % Assign to cell
        scans{i, 5} = data;
        
        % Create labels
        scans{i, 6} = repmat(scans{i, 3}, scans{i, 4}, 1); 
        
    else
        % Something is wrong with this file!
        disp(['Check file ' scans{i, 3} '.jpg'])
        continue
    end
end

% Now create prdataset from cell array
live_digits = prdataset(vertcat(scans{:, 5}), vertcat(scans{:, 6}));
live_digits = setfeatsize(live_digits, image_size);
live_digits = setname(live_digits, 'live digits');

end
