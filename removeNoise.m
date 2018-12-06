% Removes any blobs that have an area less than 50 pixels. Returns a
% cleaned image.
function [cleanedImage] = removeNoise(dirtyImage)
    cleanedImage = bwareaopen(dirtyImage, 50);
    imshowpair(dirtyImage, cleanedImage, 'montage');
end

