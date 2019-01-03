function [u_mapping, t_mapping] = get_pca(preprocessed, components, ...
    image_size)
% GET_PCA Gets PCA mapping.
%
% Inputs:
% - preprocessed: prdataset containing preprocessed data
% - components: number of components to return
% - image_size: desired size of output images

%
% Outputs:
% - u_mapping: prmapping containing the untrained PCA mapping, which must
%   be trained and then applied to the test set
% - t_mapping: prmapping containing the trained PCA mapping (for
%   visualization purposes!)
%
% Jesse Hagenaars - 02.01.2010

% Create untrained mapping
u_mapping = scalem('variance') * pcam(components);

% Create trained mapping for visualization
scaling = scalem(preprocessed, 'variance');
t_mapping = preprocessed * scaling * pcam(components);
t_mapping.size_in = [image_size 1];

end
