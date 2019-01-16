function [u_mapping, pca_vis] = get_pca(preprocessed, components, ...
    image_size)
% GET_PCA Gets PCA mapping and visualization.
%
% Inputs:
% - preprocessed: prdataset containing preprocessed data
% - components: number of components to return
% - image_size: desired size of output images
%
% Outputs:
% - u_mapping: prmapping containing the untrained PCA mapping, which must
%   be trained and then applied to the test set
% - pca_vis: prmapping containing the trained PCA mapping (only for
%   visualization purposes!)
%
% Jesse Hagenaars - 02.01.2010

% Create untrained mapping
u_mapping = scalem('variance') * pcam(components);

% Create trained mapping for visualization purposes only, since it is
%   technically not correct. Doing "pca_vis = scaling * pca_vis" would give
%   same results as using "preprocessed * u_mapping"
scaling = scalem(preprocessed, 'variance');
pca_vis = (preprocessed * scaling) * pcam(components);
pca_vis.size_in = [image_size 1];

end
