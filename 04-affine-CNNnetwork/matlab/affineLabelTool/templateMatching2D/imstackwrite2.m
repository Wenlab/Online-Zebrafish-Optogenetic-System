function imstackwrite2(path,im)
warning off
% im=single(im);
t = Tiff(path,'w');
% Ӱ���С��Ϣ��������Ƚϼ򵥣�
tagstruct.ImageLength = size(im,1); % Ӱ��ĳ���
tagstruct.ImageWidth = size(im,2);  % Ӱ��Ŀ��

% ��ɫ�ռ���ͷ�ʽ����ϸ������3.1��
tagstruct.Photometric = 1;

% ÿ�����ص���ֵλ����singleΪ�����ȸ����ͣ�����32ΪϵͳΪ32
tagstruct.BitsPerSample = 16;
% ÿ�����صĲ��θ�����һ��ͼ��Ϊ1��3�����Ƕ���ң��Ӱ����ڶ���������Գ�������3
tagstruct.SamplesPerPixel = size(im,3);
tagstruct.RowsPerStrip = 16;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
% ��ʾ����Ӱ������
tagstruct.Software = 'MATLAB'; 
% ��ʾ���������͵Ľ���
tagstruct.SampleFormat = 1;
% ����Tiff�����tag
t.setTag(tagstruct)

% ��׼����ͷ�ļ�����ʼд����
t.write(im);
% �ر�Ӱ��
t.close

end