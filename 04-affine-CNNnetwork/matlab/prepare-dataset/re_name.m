clc;
clear;
% folder='201105_fish3_h2bg6f_9dpf';
path='F:\ITK\affineTest\fishData\210924\';
filetype='.mat';

subfolders=getsubfolders(path);
fileNum=length(subfolders);

for aa=1:length(subfolders)
    folder=subfolders{aa};
    
    for i = 1:255
        disp([path,folder,'\TM_test3\','Red_' num2str(i) filetype])
        try
            oldname = [path,folder,'\TM_test3\','Red' num2str(i) filetype];
            newname = [path,folder,'\TM_test3\','Red',num2str(i,'%04d'), filetype];
            movefile(oldname,newname);
        catch ME
            disp(ME.message)
        end
        disp([path,folder,'\TM_test3\','Obj_1stAffined_' num2str(i) filetype])
        try
            oldname = [path,folder,'\TM_test3\','Obj_1stAffined_' num2str(i) filetype];
            newname = [path,folder,'\TM_test3\','Obj_1stAffined_',num2str(i,'%04d'), filetype];
            movefile(oldname,newname);
        catch ME
            disp(ME.message)
        end
        
        disp([path,folder,'\beforeRotation\','ObjRecon' num2str(i) filetype])
        try
            oldname = [path,folder,'\beforeRotation\','ObjRecon' num2str(i) filetype];
            newname = [path,folder,'\beforeRotation\','ObjRecon',num2str(i,'%04d'), filetype];
            movefile(oldname,newname);
        catch ME
            disp(ME.message)
        end
    end
end