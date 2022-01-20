savepath='F:\ITK\affineTest\fishData\210824\r20210824_X1\';
path=[savepath 'TM_test2\'];

allRotationFilenames=dir(fullfile(path,'*.tif'));

goodRotation=ones(1,length(allRotationFilenames));

for i=1:length(allRotationFilenames) 
    rotationImgName=allRotationFilenames(i).name;
    rotationImg=imstackread ([path,rotationImgName]);
    
    MIProtationImg=[max(rotationImg,[],3) squeeze(max(rotationImg,[],2));...
        squeeze(max(rotationImg,[],1))' zeros(size(rotationImg,3),size(rotationImg,3))];
    MIProtationImg=uint16(MIProtationImg); 
    
    figure;   
    imagesc(MIProtationImg)
    
    pause
    if 'a'==get(gcf,'CurrentCharacter')
        goodRotation(i)=0;
    end 
    close
end

% save([savepath,'goodRotation.mat'],'goodRotation');
