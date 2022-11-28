path='\\211.86.157.191\WenLab-Storyge\kexin\online-optogenetic-data\20220901_1619_chri-g8s-lssm_8dpf\referenceMIP.avi';

video=VideoReader(path);
writer=VideoWriter('\\211.86.157.191\WenLab-Storyge\kexin\online-optogenetic-data\20220901_1619_chri-g8s-lssm_8dpf\crop4_video_imgs\referenceMIP_crop4_flip.avi');
open(writer);

for i=14060:14690
    f=read(video,i);
    f_mip=f(:,401:800,:);
    f_mip1=flipdim(f_mip,1);
    f_mip2=flipdim(f_mip1,2);
    imshow(f_mip2)
    f(:,401:800,:)=f_mip2;
    writeVideo(writer,f)
end

close(writer)

%% 
path='\\211.86.157.191\WenLab-Storyge\kexin\online-optogenetic-data\20220901_1619_chri-g8s-lssm_8dpf\crop4_video_imgs\tracking-crop4.avi';
video=VideoReader(path);
writer=VideoWriter('\\211.86.157.191\WenLab-Storyge\kexin\online-optogenetic-data\20220901_1619_chri-g8s-lssm_8dpf\crop4_video_imgs\tracking-crop4_flip.avi');
open(writer);

for i=1:video.NumFrames
    f=read(video,i);
    f1=flipdim(f,1);
    imshow(f1)
    writeVideo(writer,f1);
end

close(writer)