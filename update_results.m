function [lowest_upd, best_cell_sizes_upd, best_comps_upd] = ...
    update_results(errors, comps, cell_size, lowest, best_cell_sizes, ...
    best_comps)
% UPDATE_RESULTS Updates the best results according to just-found error.
%
% Inputs:
% - errors: array containing classification errors
% - comps: number of PCA components
% - cell_size: HOG cell size
% - lowest: array containing lowest classification errors so far
% - best_cell_sizes: cell containing the best cell sizes so far
% - best_comps: array containing the best numbers of PCA components so far
%
% Outputs:
% - lowest_upd: updated array containing lowest classification error so far
% - best_cell_sizes_upd: updated cell containing the best cell sizes so far
% - best_comps_upd: updated array containing the best numbers of PCA
%   components so far
%
% Olle Eriksson - 25.12.2018

lowest_upd = lowest;
best_cell_sizes_upd = best_cell_sizes;
best_comps_upd = best_comps;

for j = 1:length(errors) % number classifiers
    temp = errors(j);
    if temp < lowest(j)
        lowest_upd(j) = temp;
        best_cell_sizes_upd{j} = cell_size;
        best_comps_upd(j) = comps;
    end
end

end
                