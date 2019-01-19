function [lowest_upd, best_cell_sizes_upd, best_comps_upd] = ...
    updateResults(errors, comps, cell_size, lowest, best_cell_sizes, ...
    best_comps)
% updates the best results according to just found error

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
                