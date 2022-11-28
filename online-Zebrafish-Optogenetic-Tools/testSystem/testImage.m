%% 衡量图像稳定度
path='E:\online-opto-data\20220901_1619_chri-g8s-lssm_8dpf\toRecon\raw\recon\';

allFileNames=dir(fullfile(path,'*.mat'));

errors=zeros(length(allFileNames),1);

for i=1:length(allFileNames)-1
    ObjRecon=load([path,allFileNames(i).name]);
    ObjRecon=ObjRecon.ObjRecon;
    ObjRecon(find(ObjRecon>300))=0;   %去除给光影响
    ObjReconBW = ObjRecon>nanmean((nanmean(nanmean(ObjRecon)+14)));
    
    ObjRecon2=load([path,allFileNames(i+1).name]);
    ObjRecon2=ObjRecon2.ObjRecon;
    ObjRecon2(find(ObjRecon2>300))=0;   %去除给光影响
    ObjReconBW2 = ObjRecon2>nanmean((nanmean(nanmean(ObjRecon2)+14)));
    
    err=immse(uint8(ObjReconBW),uint8(ObjReconBW2));
    errors(i)=err;
    
    disp([num2str(i) '/' num2str(length(allFileNames)-1)])
end


plot(errors)
hold on
% errors_s = smooth(errors);
% plot(errors_s)

error_idx=zeros(length(errors),1);
error_idx(find(errors>0.03))=1;


se = strel('disk',3);
error_idx = imclose(error_idx,se);
error_idx = imopen(error_idx,se);
plot(error_idx*0.1)

%% fish info
