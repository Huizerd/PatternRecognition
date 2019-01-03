function preprocessed = preprocessing(raw_data, image_size, ...
    blanks, supervised)
% PREPROCESSING Combines all preprocessing operations into one function.
%
% Inputs:
% - raw_data: prdatafile containing the raw input data
% - image_size: desired size of output images
% - blanks: number of blank rows/columns to pad to
% - supervised: whether to assign labels or not
%
% Outputs:
% - preprocessed: prdataset containing the preprocessed images
%
% Jesse Hagenaars - 02.01.2019

% Make squares, denoise, remove slant, do cropping, resize
prep_map = raw_data * im_box(blanks, 1) * filtim('remove_noise') * ...
    filtim('deslant') * im_box(blanks, 1) * im_resize(image_size);

% Convert to prdataset
prep_map = prdataset(prep_map);

% Put in empty prdataset
if supervised
    preprocessed = prdataset(prep_map.data, prep_map.labels);
else
    preprocessed = prdataset(prep_map.data);
end
    
% Set correct feature size and change name    
preprocessed = setfeatsize(preprocessed, image_size);
preprocessed = setname(preprocessed, 'preprocessed NIST');

end
