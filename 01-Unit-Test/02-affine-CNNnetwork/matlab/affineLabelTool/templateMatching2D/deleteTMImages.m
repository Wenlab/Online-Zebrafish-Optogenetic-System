function deleteTMImages(path)

%path='F:\ITK\affineTest\fishData\210824\';

    path1=[path,'\CMTKaffineResult\'];
    path2=[path,'\TM\'];


    allAffineImageName=dir(fullfile(path1,'Obj_1stAffined*.nii'));  
    allOrigImageName=dir(fullfile(path2,'*.nii'));

    allAffineImageName_No=zeros(length(allAffineImageName),1);
    for i=1:length(allAffineImageName)
    allAffineImageName_No(i,:)=str2num(allAffineImageName(i).name(16:20));
    end

    allOrigImageName_No=zeros(length(allOrigImageName),1);
    for i=1:length(allOrigImageName)
    allOrigImageName_No(i,:)=str2num(allOrigImageName(i).name(1:5));
    end


    imgdiff=setdiff(allOrigImageName_No,allAffineImageName_No);

    disp(['delete ',num2str(length(imgdiff)),' images']);
    pause

    for i=1:length(imgdiff)
        imgName=allOrigImageName(imgdiff(i)).name;
        a = find('.'==imgName);
        imname=imgName(1:a-1);
        delete([path2,imname,'.nii']);
        delete([path2,imname,'.tif']);
    end

end