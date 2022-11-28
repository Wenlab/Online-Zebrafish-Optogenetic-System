clear
path='D:\online-opto-data\20221116_1108_g8s-lssm-huc-none_8dpf\';
expLogsPath=[path,'\20221116_1108_g8s-lssm-huc-none_8dpf.txt'];
stageLogsPath=[path,'\Stage_postion2022_11_16-11_10_42.txt'];

%% 读光遗传记录
ExpLogs=readExpLogsFromTXT(expLogsPath);
laserOn=ExpLogs.laserOn;
xpos=ExpLogs.xpos;
headingAngle=ExpLogs.rotationAngleX;
laserOn(end)=0;
laserOn5 = repelem(laserOn,5);

leftLaser=zeros(length(laserOn),1);
rightLaser=zeros(length(laserOn),1);

leftLaser(find(xpos<30))=1;
leftLaser(find(laserOn==0))=0;
rightLaser(find(xpos>30))=1;
rightLaser(find(laserOn==0))=0;
leftLaser5 = repelem(leftLaser,5);
rightLaser5 = repelem(rightLaser,5);

headingAngle5=repelem(headingAngle,5);

%% 读台子移动的数据 
fid=fopen(stageLogsPath);
f=textscan(fid,'%f, x: %f, y: %f, detection: %f, head: [%f, %f], yolk: [%f, %f], confidence_h: %f, confidence_y: %f, time stamp: %f_%f_%f_%f_%f');
stage_pos=[f{2} f{3}];
head=[f{5} f{6}];
tail=[f{7} f{8}];
stage_pos=stage_pos-stage_pos(1,:);
stage_pos=[stage_pos(:,1)/10000 stage_pos(:,2)/12800];

stage_pos_1(:,1) = interp1(1:length(stage_pos),stage_pos(:,1),1:length(stage_pos)/length(laserOn5):length(stage_pos));
stage_pos_1(:,2) = interp1(1:length(stage_pos),stage_pos(:,2),1:length(stage_pos)/length(laserOn5):length(stage_pos));
head_1(:,1) = interp1(1:length(head),head(:,1),1:length(head)/length(laserOn5):length(head));
head_1(:,2) = interp1(1:length(head),head(:,2),1:length(head)/length(laserOn5):length(head));
tail_1(:,1) = interp1(1:length(tail),tail(:,1),1:length(tail)/length(laserOn5):length(tail));
tail_1(:,2) = interp1(1:length(tail),tail(:,2),1:length(tail)/length(laserOn5):length(tail));

stage_dist=zeros(length(stage_pos_1),1);
for i=1:length(stage_pos_1)
    stage_dist(i)=norm(stage_pos_1(1,:)-stage_pos_1(i,:),2);
end



%% bout检测
bouts = boutDetect(stage_pos_1);

%% 检查检测出来的bout
% % 按低倍帧数对应到340fps上，截出来视频
% load([path,'\timestamp.mat']);
% videoFolder=[path,'boutcheck\'];
% if(~exist(videoFolder))
%     mkdir(videoFolder);
% end
% 
% highFPSvideo=VideoReader([path,'\2022_10_16-15_54_44\images.avi']);
% 
% for i=1:length(bouts.start)
%     start=bouts.start(i);
%     endl=bouts.end(i);
% 
%     idx_start=find(ts340_50(:,2)==start);
%     if(isempty(idx_start))
%         for kk=1:5
%             idx_start=find(ts340_50(:,2)==start+kk);
%             if(~isempty(idx_start))
%                 break;
%             end
%         end
%     end
%     idx_end=find(ts340_50(:,2)==endl);
%     if(isempty(idx_end))
%         for kk=1:5
%             idx_end=find(ts340_50(:,2)==endl+kk);
%             if(~isempty(idx_end))
%                 break;
%             end
%         end
%     end
%     
%     videoObj=VideoWriter([videoFolder,num2str(i,'%04d'),'.avi']);
%     open(videoObj)
%     for j=idx_start:idx_end
%         f=read(highFPSvideo,j);
%         writeVideo(videoObj,f);
%     end
%     close(videoObj)
% end




%% 查看哪些bout发生在左侧打光，哪些发生在右侧打光
bout_leftLaser=zeros(length(bouts.start),1);
bout_rightLaser=zeros(length(bouts.start),1);
bout_noLazer=zeros(length(bouts.start),1);
for i=1:length(bouts.start)
    start=bouts.start(i);
    endl=bouts.end(i);
    leftLaser_inbout=leftLaser5(start:endl);
    rightLaser_inbout=rightLaser5(start:endl);
    
    if(length(find(leftLaser_inbout==1))<3&&length(find(rightLaser_inbout==1))<3)
        bout_noLazer(i)=1;
        continue;
    end

    if(length(find(leftLaser_inbout==1))>length(find(rightLaser_inbout==1)))
        bout_leftLaser(i)=1;
    else
        bout_rightLaser(i)=1;
    end
end


bouts.leftLaser=bout_leftLaser;
bouts.rightLaser=bout_rightLaser;
bouts.noLaser=bout_noLazer;

%% 鱼每个bout转过的角度
bout_angle=cell(length(bouts.start),1);
bout_sumAngle=zeros(length(bouts.start),1);
bout_angleWithStan=cell(length(bouts.start),1);
stan=[0,-180];
for i=1:length(bouts.start)
    start=bouts.start(i);
    endl=bouts.end(i);
    standard=head_1(bouts.start(i),:)-tail_1(bouts.start(i),:);
    angles=zeros(size(bouts.pos{i},1),1);
    anglewithS=zeros(size(bouts.pos{i},1),1);
    for j=start:endl
        hbt=head_1(j,:)-tail_1(j,:);
        angle = two_vector_angle(hbt,standard);
        angles(j-start+1)=angle;
        anglewithS(j-start+1)=two_vector_angle(stan,hbt);
    end
    bout_angle{i}=angles;
    bout_angleWithStan{i}=anglewithS;
    bout_sumAngle(i)=sum(diff(angles));
end
bouts.angle=bout_angle;
bouts.angleWs=bout_angleWithStan;
bouts.sumAngle=bout_sumAngle;


%% 左侧bout
% leftLaserBoutPos=bouts.pos(find(bouts.leftLaser==1));
% leftLaserBoutAngle=bouts.angleWs(find(bouts.leftLaser==1));
% for i=1:length(leftLaserBoutPos)
%     idx(i)
%     leftLaserBoutAngle{i}
%     pos=leftLaserBoutPos{i};
%     pos=pos-pos(end,:);
%     
%     pos=[pos;0,0];
% 
%     x=pos';
%     angle=mean(leftLaserBoutAngle{i}(1:3))
% 
%     angle=deg2rad(angle);
% 
%     A = [cos(angle)   -sin(angle);sin(angle)   cos(angle)];
%     y=A*x;
%     y(:,end)=NaN;
%     x(:,end)=NaN;
%     plot(x(2,:),x(1,:),'bo-')
%     hold on
%     plot(y(2,:),y(1,:),'ro-')
%     axis equal 
%     hold off
%     pause
% end


%% 右侧bout
% rightLaserBoutPos=bouts.pos(find(bouts.rightLaser==1));
% rightLaserBoutAngle=bouts.angleWs(find(bouts.rightLaser==1));
% for i=1:5
%     pos=rightLaserBoutPos{i};
%     pos=pos-pos(1,:);
%     
%     pos=[pos;0,0];
% 
%     x=pos';
%     angle=mean(rightLaserBoutAngle{i}(1))
% 
%     angle=deg2rad(angle);
% 
%     A = [cos(angle)   -sin(angle);sin(angle)   cos(angle)];
%     y=A*x;
%     y(:,end)=NaN;
%     x(:,end)=NaN;
%     plot(x(1,:),x(2,:),'b')
%     hold on
%     plot(y(1,:),y(2,:))
%     axis equal 
% %     hold off
% %     pause
% end

% angle是负的，鱼在低倍下向左转，实际为向右转（镜面对称）
% angle是正的，鱼在低倍下右转。实际为向左转
% 期望结果：
% 左侧打光时，angle为正
% 右侧打光时，angle为负
bout_leftLaser_sumAngle = bout_sumAngle(find(bout_leftLaser==1));
bout_rightLaser_sumAngle = bout_sumAngle(find(bout_rightLaser==1));
bout_noLaser_sumAngle = bout_sumAngle(find(bout_noLazer==1));
histogram(bout_leftLaser_sumAngle)
xlabel('boutRotationAngle')
ylabel('boutNum')
title('bouts during leftLaser')
savefig([path,'boutNum_leftLaser.fig'])
figure
histogram(bout_rightLaser_sumAngle)
xlabel('boutRotationAngle')
ylabel('boutNum')
title('bouts during rightLaser')
savefig([path,'boutNum_rightLaser.fig'])
figure
histogram(bout_noLaser_sumAngle)
xlabel('boutRotationAngle')
ylabel('boutNum')
title('bouts during noLaser')
savefig([path,'boutNum_noLaser.fig'])

save([path,'boutInfo.mat'],'bouts','bout_leftLaser_sumAngle', ...
    'bout_rightLaser_sumAngle','bout_noLaser_sumAngle');
%% 