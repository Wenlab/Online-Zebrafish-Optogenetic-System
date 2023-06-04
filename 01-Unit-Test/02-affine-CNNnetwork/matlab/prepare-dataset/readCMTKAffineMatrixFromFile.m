function affineMatrix=readCMTKAffineMatrixFromFile(filename)
    lineNum=1;
    fileID = fopen(filename);
    fidout=fopen('mkmatlab.txt','w');  
    while ~feof(fileID)
        tline=fgetl(fileID);
        if double(tline(1))>=45&&double(tline(1))<=57       % 判断首字符是否是数值
            fprintf(fidout,'%s\n',tline);  
        end
        lineNum=lineNum+1;
    end
    fclose(fidout);
    data=importdata('mkmatlab.txt');
    
    imgNum=length(data)/4;
    affineMatrix=zeros(imgNum,12);
    for i=1:imgNum
        if(i==1)
            affineMatrix(i,:)=[data(1:3,1)' data(1:3,2)' data(1:3,3)' data(i*4,1:3)];
        else
            affineMatrix(i,:)=[data((i-1)*4+1:(i-1)*4+3,1)' data((i-1)*4+1:(i-1)*4+3,2)' data((i-1)*4+1:(i-1)*4+3,3)' data(i*4,1:3)];
        end
    end
%     for i=1:imgNum
%         if(i==1)
%             affineMatrix(i,:)=[data(1,1:3) data(2,1:3) data(3,1:3) data(4,1:3)];
%         else
%             affineMatrix(i,:)=[data((i-1)*4+1,1:3) data((i-1)*4+2,1:3) data((i-1)*4+3,1:3) data(i*4,1:3)];
%         end
%     end

end