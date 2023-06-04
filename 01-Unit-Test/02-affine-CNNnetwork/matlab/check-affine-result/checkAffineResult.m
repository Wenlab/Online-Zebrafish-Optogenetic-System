path='C:\Users\USER\source\repos\Project3\Project3\0824_0924CMTK_AM_TM3_x10_delete\';

fixPath=[path 'fixImg\'];
movePath=[path 'movingImg\'];
resultPath=[path 'result2\'];
afterAffinePath=[path 'result\'];

allfixName=dir(fullfile(fixPath,'*.nii'));
allmoveName=dir(fullfile(movePath,'*.nii'));
allresultName=dir(fullfile(resultPath,'*.nii'));
allAffineName=dir(fullfile(afterAffinePath,'*.nii'));

goodAffine=ones(1,length(allfixName));

for i=1:length(allfixName)
    fixImgName=allfixName(i).name;
    moveImgName=allmoveName(i).name;
    resultImgName=allresultName(i).name;
    affineImgName=allAffineName(i).name;
    
    
    fixImg=niftiread([fixPath,fixImgName]);
    moveImg=niftiread([movePath,moveImgName]);
    resultImg=niftiread([resultPath,resultImgName]);
    affineImg=niftiread([afterAffinePath,affineImgName]);
    
    MIPfixImg=[max(fixImg,[],3) squeeze(max(fixImg,[],2));squeeze(max(fixImg,[],1))' zeros(size(fixImg,3),size(fixImg,3))];
    MIPfixImg=uint16(MIPfixImg);    
    
    MIPmoveImg=[max(moveImg,[],3) squeeze(max(moveImg,[],2));squeeze(max(moveImg,[],1))' zeros(size(moveImg,3),size(moveImg,3))];
    MIPmoveImg=uint16(MIPmoveImg);    
    
    MIPresultImg=[max(resultImg,[],3) squeeze(max(resultImg,[],2));squeeze(max(resultImg,[],1))' zeros(size(resultImg,3),size(resultImg,3))];
%     MIPresultImg=uint16(MIPresultImg);    

    MIPaffineResultImg=[max(affineImg,[],3) squeeze(max(affineImg,[],2));squeeze(max(affineImg,[],1))' zeros(size(affineImg,3),size(affineImg,3))];
    MIPaffineResultImg=uint16(MIPaffineResultImg);
    
    
    figure;    
    set(gcf,'outerposition',get(0,'screensize'));
    subplot(1,3,1);imshowpair(MIPfixImg,MIPmoveImg)
    xlabel('FixImg/MovingImg')
    subplot(1,3,2);imshowpair(MIPfixImg,MIPaffineResultImg)
    xlabel('FixedImg/affineImg')
    title([num2str(i) '/' num2str(length(allfixName))])
    subplot(1,3,3);imshowpair(MIPresultImg,MIPmoveImg)
    xlabel('ResultImg/movingImage')
    title([num2str(i) '/' num2str(length(allfixName))])

    pause
    if 'a'==get(gcf,'CurrentCharacter')
        goodAffine(i)=0;
    end 
    close
end

save([path,'goodAffine.mat'],'goodAffine');