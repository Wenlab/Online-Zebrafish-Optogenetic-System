path='F:\ITK\affineTest\fishData\';
savePath='F:\ITK\dataset-normMat\0824_0924CMTK_AM_TM3_x10_delete\';

fishNames=getsubfolders(path);
affineMatTraining=[];
affineMatTest=[];
affineMatVal=[];
for j=1:length(fishNames)
% for j=2:2
    fishName=fishNames{j};
    fishPath=[path,fishName,'\'];
    fishVideoNames=getsubfolders(fishPath);
    fixImgName=[fishPath,'Obj_ref2.nii'];
    for aa=1:length(fishVideoNames)
%     for aa=2:2
        fishVideoName=fishVideoNames{aa};
        fishVideoPath=[fishPath,fishVideoName,'\'];
        fishVideoPathRo=[fishPath,fishVideoName,'\TM_test3\'];

        %% read all affine matrices
        subfolders=getsubfolders(fishVideoPathRo);
        if(exist([fishVideoPathRo 'goodAffine.mat'],'file'))
            load([fishVideoPathRo 'goodAffine.mat']);
        else
            goodAffine=zeros(length(subfolders),1);
        end
%         goodAffine=ones(length(subfolders),1);
        try
            if(exist([fishVideoPath 'affineMatrix_TM3.log'],'file'))
                affineMatrices=readCMTKAffineMatrixFromFile([fishVideoPath 'affineMatrix_TM3.log']);
                affineMatrices(find(goodAffine==0),:)=[];
                affineMatrices=weightAffineMatrix3D(affineMatrices);   %% 平衡affine matrix中的参数
                deleteIdx=find(abs(affineMatrices(:,1))>100);
                affineMatrices(deleteIdx,:)=[];
            else
                disp([fishVideoPath 'affineMatrix_TM3.log not exist!!!' ]);
                continue;
            end

            %% read all fix and moving image
            allAfterAffineImageName=dir(fullfile(fishVideoPathRo,'Obj_1stAffine*.nii'));
            allAfterAffineImageName(find(goodAffine==0))=[];
            allAfterAffineImageName(deleteIdx)=[];
            allmovingImageName=dir(fullfile(fishVideoPathRo,'Red*.nii'));
            allmovingImageName(find(goodAffine==0))=[];
            allmovingImageName(deleteIdx)=[];
        catch ME
            disp(ME.message)
        end
        
        if(length(affineMatrices)~=length(allmovingImageName))
            disp('error!! affineMatrices num not euqal allmovingImage num!!! error Path:')
            disp(fishVideoPathRo)
            disp(length(affineMatrices))
            disp(length(allmovingImageName))
            continue;
        end
        if(length(allAfterAffineImageName)~=length(allmovingImageName))
            disp('error!! allFixImage num not euqal allmovingImage num!!! error Path:')
            disp(fishVideoPathRo)
            continue;
        end
        
        %% rename and write fix and moving image
        for ll=1:length(affineMatrices)
            randNum=rand(1,1);
            if(randNum>0.8)
                p=[savePath,'val','\'];
                affineMatVal=cat(1,affineMatVal,affineMatrices(ll,:));
            elseif(randNum>0.7&&randNum<0.8)
                p=[savePath,'test','\'];
                affineMatTest=cat(1,affineMatTest,affineMatrices(ll,:));
            else
                p=[savePath,'train','\'];
                affineMatTraining=cat(1,affineMatTraining,affineMatrices(ll,:));
            end
            if ~exist(p,'dir')
                mkdir(p);
            end
            if ~exist([p,'fixImg\'],'dir')
                mkdir([p,'fixImg\']);
            end
            if ~exist([p,'movingImg\'],'dir')
                mkdir([p,'movingImg\']);
            end
            if ~exist([p,'afterAffine\'],'dir')
                mkdir([p,'afterAffine\']);
            end
            try
                movingname=allmovingImageName(ll).name;
                afterAffineName=allAfterAffineImageName(ll).name;
                disp([fishVideoPathRo,movingname])
%                 disp([fishVideoPathRo,afterAffineName])
                copyfile(fixImgName,[p,'fixImg\',fishName,'_',num2str(aa,'%04d'),'_',num2str(ll,'%04d'),'.nii'])
                copyfile([fishVideoPathRo,movingname],[p,'movingImg\',fishName,'_',num2str(aa,'%04d'),'_',num2str(ll,'%04d'),'.nii'])
                copyfile([fishVideoPathRo,afterAffineName],[p,'afterAffine\',fishName,'_',num2str(aa,'%04d'),'_',num2str(ll,'%04d'),'.nii'])
%                 imstack=imstackread([fishVideoPathRo,movingname]);
%                 savenitfi(imstack,[p,'movingImg\',fishVideoName,'_',num2str(ll,'%04d'),'.nii'])
%                 disp([fishVideoPathRo,movingname])
            catch ME
                disp(ME.message)
            end
        end
    end
end


%% write affine matrix to .csv
writematrix(affineMatTraining,[[savePath,'train','\'],'affineMatrices.csv']);
writematrix(affineMatVal,[[savePath,'val','\'],'affineMatrices.csv']);
writematrix(affineMatTest,[[savePath,'test','\'],'affineMatrices.csv']);