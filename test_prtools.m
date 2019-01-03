% Test file PRTools pcam/scalem
%
% Jesse Hagenaars - 02.01.2019

clear; close all

raw_data = prnist(0:9, 1:100:1000);

% No deslanting, deslant without binarize, deslant wit binarize
data = raw_data * im_box([5 5 5 5], 1) * im_resize([50 50]);
data2 = raw_data * im_box([5 5 5 5], 1) * filtim('deslant') * im_resize([50 50]);
data3 = raw_data * im_box([5 5 5 5], 1) * filtim('deslant') * filtim('imbinarize', {'global'}) * im_resize([50 50]);

data = prdataset(data);
data2 = prdataset(data2);
data3 = prdataset(data3);

% No deslant
mapping1 = data * (scalem('variance') * pcam(50));
mapping1.size_in = [50 50 1];
figure
show(mapping1)

% Deslant without binarize
mapping2 = data2 * (scalem('variance') * pcam(50));
mapping2.size_in = [50 50 1];
figure
show(mapping2)

% Deslant with binarize
mapping3 = data3 * (scalem('variance') * pcam(50));
mapping3.size_in = [50 50 1];
figure
show(mapping3)