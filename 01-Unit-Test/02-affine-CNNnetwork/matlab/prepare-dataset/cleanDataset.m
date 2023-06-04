path='F:\ITK\affineTest\fishData\210924\r20210924_2_X31\TM_test3\';
savepath=path;

fixImgName=[path,'Obj_ref2.nii'];
fixImage = niftiread(fixImgName);
MIPfixImage=[max(fixImage,[],3) squeeze(max(fixImage,[],2));squeeze(max(fixImage,[],1))' zeros(size(fixImage,3),size(fixImage,3))];
MIPfixImage=uint16(MIPfixImage);


allAffineImageName=dir(fullfile(path,'Obj_1stAffined*.nii'));  
allOrigImageName=dir(fullfile(path,'Red*.tif'));
goodAffine=ones(length(allAffineImageName),1);
                         

% f=figure(1);
% f.WindowState = 'maximized';
for i=1:255
    try
        AffineImage=niftiread([path,'Obj_1stAffined_' num2str(i,'%04d') '.nii']);
        disp([path,'Obj_1stAffined_' num2str(i,'%04d') '.nii']);
        OrigImg=imstackread([path,'Red' num2str(i) '.tif']);
        OrigImg=imstack2nifti2(OrigImg);
    catch ME
        disp(ME.message)
    end
    MIPAffineImage=[max(AffineImage,[],3) squeeze(max(AffineImage,[],2));squeeze(max(AffineImage,[],1))' zeros(size(AffineImage,3),size(AffineImage,3))];
    MIPAffineImage=uint16(MIPAffineImage);    
    MIPOrigImg=[max(OrigImg,[],3) squeeze(max(OrigImg,[],2));squeeze(max(OrigImg,[],1))' zeros(size(OrigImg,3),size(OrigImg,3))];
    MIPOrigImg=uint16(MIPOrigImg);   
    
    figure;    
    set(gcf,'outerposition',get(0,'screensize'));
    subplot(1,2,1);imshowpair(MIPOrigImg,MIPfixImage)
    xlabel('OrigImg/FixedImg')
    subplot(1,2,2);imshowpair(MIPAffineImage,MIPfixImage)
    xlabel('AffineImg/FixedImg')
    title([num2str(i) '/' num2str(length(allAffineImageName))])

    pause
    if 'a'==get(gcf,'CurrentCharacter')&&'s'==get(gcf,'CurrentCharacter')
        goodAffine(i)=0;
    end 
    close
end

save([savepath,'goodAffine.mat'],'goodAffine');