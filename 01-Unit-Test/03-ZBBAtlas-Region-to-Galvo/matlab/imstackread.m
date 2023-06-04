function imstack=imstackread(filename)

info=imfinfo(filename);
imstack=zeros(info(1).Height,info(1).Width,size(info,1));

if info(1).BitDepth==8
    imstack=uint8(imstack);
else
    imstack=uint16(imstack);
end

for ii=1:size(info)
    imstack(:,:,ii)=imread(filename,'Info',info(ii));
end
disp(['load: ' filename ': finished']);
end