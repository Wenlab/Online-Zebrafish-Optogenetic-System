% [Function lookuparea4.m]
% Purpose:
%   to look up the region in a specific area on the atlas
% Input:
%         queryRegion,  the region number to be checked
%         A3,           footprint of ROIs
%         tbl,          the table of each pixel number to brain areas
%
% Output:
%         areaName,     the names of the brain area, in cell array (maybe empty or multiple)
%         perc,         the percentage of how much pixels in the area  
%
% 
% Edited by Chen Shen, July 20 2021
 


function [areaName,perc] = lookuparea4(queryRegion, A3, tbl)


pixelIdx = find(A3(:,queryRegion));
areaName = arrayfun(@(i) tbl.(1)((tbl.(2)==i)),pixelIdx,'un',0);
areaName = cat(1,areaName{:});
[areaName,ia,ic] = unique(areaName);

%areaName=cell2mat(areaName);
A = accumarray(ic,1);
perc = A/sum(A);

if isempty(perc)
    perc = 0;
end
