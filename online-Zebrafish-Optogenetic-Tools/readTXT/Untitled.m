path='\\211.86.157.191\WenLab-Storyge\kexin\online-optogenetic-data\20220920_1003_g8s-lssm-tph2-chriR_7dpf\20220920_1003_g8s-lssm-tph2-chriR_7dpf.txt';
ExpLogs=readExpLogsFromTXT(path);

videoPath='E:\online-opto-data\20220901_1619_chri-g8s-lssm_8dpf\referenceMIP.avi';

vid= VideoReader(videoPath);
vwriter = VideoWriter('E:\online-opto-data\20220901_1619_chri-g8s-lssm_8dpf\referenceMIP_1.avi');
open(vwriter);

frameNum=vid.NumFrames;


for i=1:frameNum
    frame = read(vid,i);
    laser=ExpLogs.laserOn(i);
    
    imshow(frame,'border','tight','initialmagnification','fit')
    if(laser==1)
        txt = 'LaserOn';
        t = text(400,20,txt,'Fontsize',20,'Color','r');
%     else
%         txt = 'galvoOff';
%         t = text(400,20,txt,'Fontsize',20,'Color','r');
    end
    
    [m,n,q]=size(frame);
    set (gcf,'Position',[0,0,n,m])
    axis normal;
    f=getframe(figure(1));
    writeVideo(vwriter,f);
    
%     close all
    
end

close(vwriter)


%% 
laserOn=ExpLogs.laserOn;
xsize=ExpLogs.xsize;
xpos=ExpLogs.xpos;
ysize=ExpLogs.ysize;
ypos=ExpLogs.ypos;
rotationAngleX=ExpLogs.rotationAngleX;
rotationAngleY=ExpLogs.rotationAngleY;
cropPoint=ExpLogs.cropPoint;

cropStart=15680;
cropEnd=16210;

laserOn=laserOn(cropStart:cropEnd);
xsize=xsize(cropStart:cropEnd);
xpos=xpos(cropStart:cropEnd);
ysize=ysize(cropStart:cropEnd);
ypos=ypos(cropStart:cropEnd);
rotationAngleX=rotationAngleX(cropStart:cropEnd);
rotationAngleY=rotationAngleY(cropStart:cropEnd);
cropPoint=cropPoint(cropStart:cropEnd,:);