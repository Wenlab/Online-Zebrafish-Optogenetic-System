%% read valvo voltage and image
path='\\211.86.157.191\WenLab-Storyge\kexin\calibrationData\20220727_1543\';
folders=getsubfolders(path);

galvoVoltagePairs=zeros(length(folders),2);
galvoPointsLoc=zeros(length(folders),2);

for i=1:length(folders)
    num_str = regexp(folders{i},'\-?\d*\.?\d*','match');
    voltagePair = str2double(num_str);
    if length(voltagePair)>2
        continue
    end
    
    path_MIP=[path,folders{i},'\MIP\'];
    path_recon=[path,folders{i},'\recon\'];


    imgNames=dir(fullfile(path_recon,'*.mat'));
    meanLoc=zeros(length(imgNames),2);
    for j=1:length(imgNames)
        imgname=imgNames(j).name;
        load([path_recon,imgname]);

        img_center=ObjRecon(:,:,25);
        imgMax=max(max(img_center));
        if(imgMax<1000)
            continue;
        end
        [x,y]=find(img_center==imgMax);

        imgMaxLoc=[x,y];
        meanLoc(j,:)=imgMaxLoc;
    end
    meanLoc=mean(meanLoc,1);
    galvoPointsLoc(i,:)=meanLoc;
    galvoVoltagePairs(i,:)=voltagePair;
    
    clc
    disp([num2str(i),'/',num2str(length(folders))])
end

%% generate calibration map
% check
map=zeros(200,200);
for i=1:length(galvoPointsLoc)
    if(sum(galvoPointsLoc(i,:))>0)
        map(round(galvoPointsLoc(i,1)),round(galvoPointsLoc(i,2)))=255;
    end
end
imshow(map)

galvoPointsLoc2=galvoPointsLoc;
galvoVoltagePairs2=galvoVoltagePairs;
idx=find(galvoPointsLoc(:,1)==0);

galvoPointsLoc2(idx,:)=[];
galvoVoltagePairs2(idx,:)=[];

galvoVoltagePairs2=galvoVoltagePairs2(~any(isnan(galvoPointsLoc2),2),:);
galvoPointsLoc2=galvoPointsLoc2(~any(isnan(galvoPointsLoc2),2),:);

tform=estimateGeometricTransform(galvoPointsLoc2,galvoVoltagePairs2,'affine');

CCDPoint_t=zeros(200*200, 2);
k=1;
for i=0:(200-1)    
    for j=0:(200-1)
        point=[i j];
        CCDPoint_t(k,:)=point;
        k=k+1;
    end
end

[a,b]=size(CCDPoint_t);
CCDPoint_t = [CCDPoint_t ones(a,1)];
GalvoPoint_t=CCDPoint_t*tform.T;
GalvoPoint_tX=GalvoPoint_t(:,1);
GalvoPoint_tY=GalvoPoint_t(:,2);        
GalvoX_matrix=zeros(200,200);
GalvoY_matrix=zeros(200,200);

k=1;
for i=1:200
    for j=1:200
        GalvoX_matrix(i,j)=GalvoPoint_tX(k);
        GalvoY_matrix(i,j)=GalvoPoint_tY(k);
        k=k+1;
    end
end

GalvoX_matrix=GalvoX_matrix';
GalvoY_matrix=GalvoY_matrix';
matlab2opencv(GalvoX_matrix,[path,'GalvoX_matrix.yaml'],'w','f');
matlab2opencv(GalvoY_matrix,[path,'GalvoY_matrix.yaml'],'w','f');
