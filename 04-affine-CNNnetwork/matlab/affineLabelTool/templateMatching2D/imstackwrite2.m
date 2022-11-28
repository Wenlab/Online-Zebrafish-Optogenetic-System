function imstackwrite2(path,im)
warning off
% im=single(im);
t = Tiff(path,'w');
% 影像大小信息（这两项比较简单）
tagstruct.ImageLength = size(im,1); % 影像的长度
tagstruct.ImageWidth = size(im,2);  % 影像的宽度

% 颜色空间解释方式，详细见下文3.1节
tagstruct.Photometric = 1;

% 每个像素的数值位数，single为单精度浮点型，对于32为系统为32
tagstruct.BitsPerSample = 16;
% 每个像素的波段个数，一般图像为1或3，但是对于遥感影像存在多个波段所以常常大于3
tagstruct.SamplesPerPixel = size(im,3);
tagstruct.RowsPerStrip = 16;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
% 表示生成影像的软件
tagstruct.Software = 'MATLAB'; 
% 表示对数据类型的解释
tagstruct.SampleFormat = 1;
% 设置Tiff对象的tag
t.setTag(tagstruct)

% 以准备好头文件，开始写数据
t.write(im);
% 关闭影像
t.close

end