%% 制作模板
template=imstackread('F:\ITK\affineTest\fishData\freemoving0824\r20210824_X9\afterRotation\Red61.tif');
path='F:\ITK\affineTest\fishData\freemoving0824\r20210824_X7\beforeRotation\';
save_path='F:\ITK\affineTest\fishData\freemoving0824\r20210824_X7\TM_test\';
if(~exist(save_path,'dir'))
    mkdir(save_path)
end

rotateAngle_z=0:5:355;  % 绕z轴的旋转
rotateAngle_x=-1:1:1;  %绕x轴的旋转
rotateAngle_y=-1:1:1;  %绕y轴的旋转

% templateBW = template>nanmean((nanmean(nanmean(template)+4)));
thre=30;
templateBW = template>thre;

template_ro=cell([length(rotateAngle_z),length(rotateAngle_x),length(rotateAngle_y)]);
centers=cell([length(rotateAngle_z),length(rotateAngle_x),length(rotateAngle_y)]);
shift=cell([length(rotateAngle_z),length(rotateAngle_x),length(rotateAngle_y)]);
for i=1:length(rotateAngle_z)
    for j=1:length(rotateAngle_x)
        for k=1:length(rotateAngle_y)
            t=imrotate3(template,rotateAngle_z(i),[0 0 1]);
            t=imrotate3(t,rotateAngle_x(j),[1 0 0],'crop');
            t=imrotate3(t,rotateAngle_y(k),[0 1 0],'crop');
            t_BW = t>thre;
            stats = regionprops3(t_BW, 'Centroid','Volume','BoundingBox');
            propX = cell2mat(table2cell(stats));
            [maxv, index]=max(propX(:,1));
            CentroID = round(propX(index,2:4));
            bbox=round(propX(index,5:10));
            shift{i,j,k}=[-CentroID size(t)-CentroID];
            centers{i,j,k}=CentroID;
            template_ro{i,j,k}=t_BW;
        end
    end 
end

%% 读取图像并匹配
allImageName=dir(fullfile(path,'stackr*.tif'));
for aa=1:length(allImageName)
    image=imstackread([path,allImageName(aa).name]);
    imageBW=image>thre;
    stats=regionprops3(imageBW, 'Centroid','Volume','BoundingBox');
    propX = cell2mat(table2cell(stats));
    if(size(propX,1)<1)
        continue
    end
    [maxv, index]=max(propX(:,1));
    imageCentroID = round(propX(index,2:4));
    bBox=round(propX(index,5:10));
    
    try
        imageROI=image(imageCentroID(2)-35:imageCentroID(2)+35-1, imageCentroID(1)-35:imageCentroID(1)+35-1, 1:50);
    catch ME
        disp(ME.message)
    end
    
    tic
    err=zeros(size(template_ro,1),size(template_ro,2),size(template_ro,3));
%     ncc=zeros(size(template_ro,1),size(template_ro,2),size(template_ro,3));
    for i=1:size(template_ro,1)
        for j=1:size(template_ro,2)
            for k=1:size(template_ro,3)
                temp=template_ro{i,j,k};
                s=shift{i,j,k};
                c=centers{i,j,k};
               % imageROI=image(imageCentroID(1)+s(1):imageCentroID(1)+s(4),imageCentroID(2)+s(2):imageCentroID(2)+s(5),:);
                tempROI=temp(c(2)-35:c(2)+35-1, c(1)-35:c(1)+35-1, 1:50);
               % imageROI_resize=imresize3(imageROI,size(temp));
                imageROIBW=imageROI>thre;
                err(i,j,k)=immse(uint8(tempROI),uint8(imageROIBW));
%                 [~,I_NCC,~]=template_matching(uint8(tempROI),uint8(imageROIBW));
%                 ncc(i,j,k)=max(I_NCC,[],'all');
            end
        end
    end
    toc
    idx=find(err==(min(err,[],'all')));
    [z,x,y] = ind2sub(size(err), idx);
    z=z(1);
    x=x(1);
    y=y(1);
    figure(1)
    imshow(sum(template_ro{z,x,y},3));
    title('template_mse')
    
%     idx=find(ncc==(max(ncc,[],'all')));
%     [a,b,c] = ind2sub(size(err), idx);
%     figure(6)
%     imshow(sum(template_ro{a,b,c},3));
%     title('template_ncc')
    
    MIP=[max(imageROI,[],3) squeeze(max(imageROI,[],2));squeeze(max(imageROI,[],1))' zeros(size(imageROI,3),size(imageROI,3))];
        MIP=uint16(MIP);
    figure(2); imagesc(MIP);  
    title('ROIBeforeRotation')
    
    figure(3)
    imageROIBW=imageROI>thre;
    imshow(sum(imageROIBW,3))
    title('BWROIBeforeRotation')

    %% 将图像旋转至模板的角度
    roZ=rotateAngle_z(z);
    roX=rotateAngle_x(x);
    roY=rotateAngle_y(y);
    minerror_mse=min(err,[],'all');

    image=imrotate3(image,-roZ,[0 0 1], 'crop');
    image=imrotate3(image,-roX,[1 0 0], 'crop');
    image=imrotate3(image,-roY,[0 1 0], 'crop');
    
    
    
    ObjReconRedBW = image>nanmean((nanmean(nanmean(image)+4)));
    statsX = regionprops3(ObjReconRedBW,'volume','Centroid');
    propX = cell2mat(table2cell(statsX));
    [maxv, index]=max(propX(:,1));
    CentroID = round(propX(index,2:4));
    % 4x4 binning
    if CentroID(2) < (100)
        if (CentroID(2)-37)>0 && (CentroID(1)-35)>0 && (CentroID(2)+61) < 600 && (CentroID(1) + 35) < 150 && (CentroID(2) -37 ) < 150 && (CentroID(1)-35 < 150)

        image = image(CentroID(2)-35:CentroID(2)+61, CentroID(1)-35:CentroID(1)+35, 1:50);
        image = imrotate(image, 180,'bicubic', 'crop');
        image = flip(image, 3);
        image = flip(image, 2);
        end
    else
%         image = imrotate(image, 180,'bicubic', 'crop');
        image = flip(image, 3);
%         image = flip(image, 2);
        image = image(CentroID(2)-61:CentroID(2)+32, CentroID(1)-32:CentroID(1)+32, 1:50);
    end 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Rescale to zbb

    [XObj YObj ZObj] = size(image);

    % size of reference atlas
    XRef = 95; YRef = 77; ZRef = 52;

%     [X,Y,Z] = meshgrid(1:YObj/YRef:YObj+1,1:XObj/XRef:XObj+1,1:ZObj/ZRef:ZObj+1);
%     RescaledRed = uint16(interp3(image,X,Y,Z,'cubic'));
%     RescaledRed = RescaledRed(1:XRef, 1:YRef, 1:ZRef);
    RescaledRed=imresize3(image,[XRef YRef ZRef]);
    
    MIPC=[max(RescaledRed,[],3) squeeze(max(RescaledRed,[],2));squeeze(max(RescaledRed,[],1))' zeros(size(RescaledRed,3),size(RescaledRed,3))];
        MIPC=uint16(MIPC);
    figure(4); imagesc(MIPC);  
    title('ROIAfterRotation')
    imwrite(uint16(MIPC),[save_path 'MIP' '_' num2str(aa) '.tif']);
    
    savenitfi(RescaledRed,[save_path,'Red_TM',num2str(aa),'.nii']);
    imstackwrite([save_path,'Red_TM',num2str(aa),'.tif'],RescaledRed);
    
%     pause
end

%% test template
% figure(1)
% for i=1:size(template_ro,1)
%     for j=1:size(template_ro,2)
%         for k=1:size(template_ro,3)
%             img=template_ro{i,j,k};
%             img=sum(img,3);
%             imagesc(img);
%             pause
%         end
%     end
% end