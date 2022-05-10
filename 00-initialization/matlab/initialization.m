debug=1;
% 在鱼的追踪稳定后，用andor直接拍100张图像，并从中选取10张成像效果好的图
% 可保存成连续或分散的tif
% 重构

prompt = 'Please input fish name: ';
str = input(prompt,'s');
if isempty(str)
    str = 'fish';
end

% 测试时先使用重构好的10张图像
path='F:\ITK\initialization\matlab\standard_test\';
savepath=['F:\ITK\initialization\matlab\',str,'\'];
if ~exist(savepath,'dir')
    mkdir(savepath)
end

% 求平均图像
fileNames=dir(fullfile(path,'*.tif'));

meanImage=uint16(zeros(200,200,50,length(fileNames)));
for i=1:length(fileNames)
    img=imstackread(fullfile(path,fileNames(i).name));
    meanImage(:,:,:,i)=img;
end

meanImage=uint16(mean(meanImage,4));
save([savepath,'template.mat'],'meanImage');

MIPC=[max(meanImage,[],3) squeeze(max(meanImage,[],2));squeeze(max(meanImage,[],1))' zeros(size(meanImage,3),size(meanImage,3))];
MIPC=uint16(MIPC);
 figure(1); imagesc(MIPC);  
afterRotation=rotation3DFishImg(meanImage);

%% xy平面的模板
template=afterRotation;
templateXY=max(template,[],3);
rotationAngleXY=0:1:359;
BWtemplateXY2D = templateXY>nanmean((nanmean(nanmean(templateXY)+4)));
template_roXY=cell(length(rotationAngleXY),1);
if ~exist([savepath,'templateXY\'],'dir')
    mkdir([savepath,'templateXY\'])
end
for i=1:length(rotationAngleXY)
    t=imrotate(BWtemplateXY2D,rotationAngleXY(i),'crop');
    template_roXY{i}=t;
    imstackwrite([savepath,'templateXY\','templateXY',num2str(i),'.tif'],t);
end

%% yz平面的模板
templateYZ=squeeze(max(template,[],2));
BWtemplateYZ2D = templateYZ>nanmean((nanmean(nanmean(templateYZ)+14)));
rotationAngleYZ=-15:1:15;
template_roYZ=cell(length(rotationAngleYZ),1);
if ~exist([savepath,'templateYZ\'],'dir')
    mkdir([savepath,'templateYZ\'])
end
for i=1:length(template_roYZ)
    t=imrotate(BWtemplateYZ2D,rotationAngleYZ(i),'crop');
    template_roYZ{i}=t;
    imstackwrite([savepath,'templateYZ\','templateYZ',num2str(i),'.tif'],t);
end

%% segmentation
BWObjRecon = afterRotation>nanmean((nanmean(nanmean(afterRotation)+4)));
idx=find(BWObjRecon==1);
[x,y,z] = ind2sub(size(BWObjRecon), idx);
pos=[y,x,z];
CentroID=round(mean(pos,1));
% crop out
try
    ObjReconRed = afterRotation(CentroID(2)-60:CentroID(2)+34, CentroID(1)-37:CentroID(1)+38, 1:50);
catch ME
    disp(ME.message)
    ObjReconRed=zeros(95,70,50);
end
if(debug)
    MIPC=[max(ObjReconRed,[],3) squeeze(max(ObjReconRed,[],2));squeeze(max(ObjReconRed,[],1))' zeros(size(ObjReconRed,3),size(ObjReconRed,3))];
    MIPC=uint16(MIPC);
    figure(6); imagesc(MIPC);  
end

%     % rescale to zbb
[XObj YObj ZObj] = size(ObjReconRed);
% size of reference atlas
XRef = 95; YRef = 77; ZRef = 52;
[X,Y,Z] = meshgrid(1:YObj/YRef:YObj+1,1:XObj/XRef:XObj+1,1:ZObj/ZRef:ZObj+1);
RescaledRed = uint16(interp3(double(ObjReconRed),X,Y,Z,'cubic'));
RescaledRed = RescaledRed(1:XRef, 1:YRef, 1:ZRef);

% imstackwrite([savepath,'toAffineWithZBB.tif'],RescaledRed);
savenitfi(RescaledRed,[savepath,'toAffineWithZBB.nii']);
MIP=max(RescaledRed,[],3);
imwrite(MIP,[savepath,'templateMIP.png']);

zbb_rescale=niftiread('zbb_rescale.nii');
savenitfi(zbb_rescale,[savepath,'zbb_rescale.nii']);

pause
close all;