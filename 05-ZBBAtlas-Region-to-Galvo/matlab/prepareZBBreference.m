path='F:\ITK\queryRegions\ZBB anatomy\anatomy_reference\';
allAreaName=dir(fullfile(path,'*.tif'));


areaList=[];
for i=1:length(allAreaName)
    areaName=allAreaName(i).name;
    areaStack=imstackread([path,areaName]);
    areaStack=areaStack(:,1:380,:);
    
    areaStack=imresize3(areaStack,1/4,'nearest');
    areaStack=areaStack(1:77,1:95,1:52);
    
    areaPixel=find(areaStack==255);
    [row,col,channel]=ind2sub([77,95,52],areaPixel);
    areaPos=[row,col,channel];
    
    areaLabel=string(areaName(1:end-4));
    areaLabel=repmat(areaLabel,[size(areaPos,1),1]);
    
    areaLabelPair=cat(2,areaLabel,areaPos);
    areaList=cat(1,areaList,areaLabelPair);
end

save([path,'anatomyList_4bin.mat'],'areaList');
writematrix(areaList,[path,'anatomyList_4bin.txt'])