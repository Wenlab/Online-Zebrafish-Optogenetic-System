debug=0;

%% make x-y 2D template
% load('F:\ITK\affineTest\fishData\210924\r20210924_2_X6\beforeRotation\ObjRecon0250.mat');
load('/home/data2/backup_customer/kexin/fishData/210924/recon/r20210924_2_X6_ObjRecon_0250.mat');
rootPath='/home/data2/backup_customer/kexin/fishData/210924/';



template=ObjRecon;
template=rotation3DFishImg(template);
templateXY=max(template,[],3);
rotationAngleXY=0:1:359;
BWtemplateXY2D = templateXY>nanmean((nanmean(nanmean(templateXY)+4)));
template_roXY=cell(length(rotationAngleXY),1);
for i=1:length(rotationAngleXY)
    t=imrotate(BWtemplateXY2D,rotationAngleXY(i),'crop');
    template_roXY{i}=t;
end

%% make y-z 2D template
templateYZ=squeeze(max(template,[],2));
BWtemplateYZ2D = templateYZ>nanmean((nanmean(nanmean(templateYZ)+14)));

rotationAngleYZ=-15:1:15;
template_roYZ=cell(length(rotationAngleYZ),1);
for i=1:length(template_roYZ)
    t=imrotate(BWtemplateYZ2D,rotationAngleYZ(i),'crop');
    template_roYZ{i}=t;
%     imshow(t)
end

%% read image and matching
reconPath=[rootPath,'/recon/'];
TM_Path=[rootPath,'/TM/'];
if ~exist(TM_Path,'dir')
    mkdir(TM_Path);
end
allImageName=dir(fullfile(reconPath,'*.mat'))
%% write out rotation angle and crop 
rotationAngleXY2=zeros(length(allImageName),1);
rotationAngleYZ2=zeros(length(allImageName),1);
cropPoint=zeros(length(allImageName),2);
rescale=zeros(length(allImageName),3);
for kk=1:length(allImageName)
    imgname=allImageName(kk).name;
    a = find('.'==imgname);
    iname=imgname(1:a-1);

    disp([reconPath imgname])
    ObjRecon=load([reconPath imgname]);
    ObjRecon=ObjRecon.ObjRecon;
    %% X-Y rotation
    image2D_XY=max(ObjRecon,[],3);
    img2DBW_XY=image2D_XY>nanmean((nanmean(nanmean(image2D_XY))));
    err_XY=zeros(length(rotationAngleXY),1);

    for i=1:length(rotationAngleXY)
        temp=template_roXY{i};
        err_XY(i)=immse(uint8(temp),uint8(img2DBW_XY));
    end

    idx=find(err_XY==(min(err_XY,[],'all')));
    idx=idx(1);
    imageRotated3D=imrotate3(ObjRecon,-rotationAngleXY(idx),[0 0 1], 'crop');
    rotationAngleXY2(kk)=-rotationAngleXY(idx);
    
    %% Y-Z rotation
%         image2D_YZ=squeeze(max(imageRotated3D,[],2));
%         img2DBW_YZ=image2D_YZ>nanmean((nanmean(nanmean(image2D_YZ)+14)));
%         if(debug)
%             figure(5)
%             imshow(img2DBW_YZ)
%         end
%         err_YZ=zeros(length(rotationAngleYZ),1);
%         for i=1:length(rotationAngleYZ)
%             temp=template_roYZ{i};
%             err_YZ(i)=immse(uint8(temp),uint8(img2DBW_YZ));
%         end
%         idx2=find(err_YZ==(min(err_YZ,[],'all')));
%         idx2=idx2(1);
%         imageRotated3D=imrotate3(imageRotated3D,rotationAngleYZ(idx2),[1 0 0], 'crop');
%         rotationAngleYZ2(kk)=rotationAngleYZ(idx2);
%         
%         if(debug)
%             figure(1)
%             imshow(template_roYZ{idx2})
%             disp(-rotationAngleYZ(idx2))
%         end
    
%         MIPC=[max(imageRotated3D,[],3) squeeze(max(imageRotated3D,[],2));squeeze(max(imageRotated3D,[],1))' zeros(size(imageRotated3D,3),size(imageRotated3D,3))];
%         MIPC=uint16(MIPC);
%         figure(4); imagesc(MIPC);  
%         pause
%         
    %% segmentation
    BWObjRecon = imageRotated3D>nanmean((nanmean(nanmean(imageRotated3D)+4)));
    idx=find(BWObjRecon==1);
    [x,y,z] = ind2sub(size(BWObjRecon), idx);
    pos=[y,x,z];
    CentroID=round(mean(pos,1));
    % crop out
    try
        ObjReconRed = imageRotated3D(CentroID(2)-60:CentroID(2)+34, CentroID(1)-37:CentroID(1)+38, 1:50);
        cropPoint(kk,:)=[CentroID(2)-61 CentroID(1)-35];
    catch ME
        disp(ME.message)
        ObjReconRed=zeros(95,70,50);
    end
    if(debug)
        MIPC=[max(ObjReconRed,[],3) squeeze(max(ObjReconRed,[],2));squeeze(max(ObjReconRed,[],1))' zeros(size(ObjReconRed,3),size(ObjReconRed,3))];
        MIPC=uint16(MIPC);
        figure(6); imagesc(MIPC);  
        pause
    end

%     % rescale to zbb
    [XObj YObj ZObj] = size(ObjReconRed);
    % size of reference atlas
    XRef = 95; YRef = 76; ZRef = 50;
    [X,Y,Z] = meshgrid(1:YObj/YRef:YObj+1,1:XObj/XRef:XObj+1,1:ZObj/ZRef:ZObj+1);
    RescaledRed = uint16(interp3(double(ObjReconRed),X,Y,Z,'cubic'));
    RescaledRed = RescaledRed(1:XRef, 1:YRef, 1:ZRef);
    rescale(kk,:)=[XObj/XRef YObj/YRef ZObj/ZRef];

    savenitfi(RescaledRed,[TM_Path,num2str(kk,'%05d'),'_',iname,'_TMcrop','.nii']);
    imstackwrite2([TM_Path,num2str(kk,'%05d'),'_',iname,'_TMcrop','.tif'],RescaledRed);
    
    saveTMparams(TM_Path,rotationAngleXY2,rotationAngleYZ2,cropPoint,rescale)
end

