path='E:\online-opto-data\20220817_1427_t_tdpf\20220817_1427_t_tdpf.txt';
fid=fopen(path);
frameNum=[];
laserOn=[];
xsize=[];
xpos=[];
ysize=[];
ypos=[];
rotationAngleX=[];
rotationAngleY=[];
cropPoint=[];
Moving2FixAffineMatrix=[];
while ~feof(fid)
    [fieldName,value]=read_a_line(fid);
    if(~isempty(fieldName))
        switch fieldName
            case 'frameNum'
                frameNum = cat(1,frameNum,str2num(value));
            case 'laserOn'
                laserOn = cat(1,laserOn,str2num(value));
            case 'xsize'
                xsize = cat(1,xsize,str2num(value));
            case 'xpos'
                xpos = cat(1,xpos,str2num(value));
            case 'ysize'
                ysize = cat(1,ysize,str2num(value));
            case 'ypos'
                ypos = cat(1,ypos,str2num(value));
            case 'rotationAngleX'
                rotationAngleX = cat(1,rotationAngleX,str2num(value));
            case 'rotationAngleY'
                rotationAngleY = cat(1,rotationAngleY,str2num(value));
            case 'cropPoint'
                cropPoint = cat(1,cropPoint,str2num(value));
            case 'Moving2FixAffineMatrix'
                Moving2FixAffineMatrix = cat(1,Moving2FixAffineMatrix,str2num(value));
        end
    end
end
fclose(fid);
disp(['load: ' path ': finished']);






