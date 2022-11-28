%% 220930下午 0.3mW弱蓝光常亮
clear
path='E:\online-opto-data\data-to-analysis\20220930_1655_g8s-lssm-tph2-chriR_10dpf\20220930_1655_g8s-lssm-tph2-chriR_10dpf.txt';
ExpLogs=readExpLogsFromTXT(path);
laserOn=ExpLogs.laserOn;
laserOn(end)=0;
headingAngle=ExpLogs.rotationAngleX;

blueOn=zeros(length(laserOn),1);
blueOn(3138:12622)=1;
blueOn(15676:25168)=1;
blueOn(28244:end)=1;
blueOn(end)=0;

figure
plot(headingAngle);
hold on
drawShadow(blueOn,360,'b');
drawShadow(laserOn,360,'y');
xlabel('frames')
ylabel('headingAngle')
hold off

blueOn5 = repelem(blueOn,5);
laserOn5 = repelem(laserOn,5);

% 台子的移动距离
path='E:\online-opto-data\data-to-analysis\20220930_1655_g8s-lssm-tph2-chriR_10dpf\Stage_postion2022_9_30-17_0_14.txt';
fid=fopen(path);
f=textscan(fid,'%f, x: %f, y: %f, detection: %f, head: [%f, %f], yolk: [%f, %f], confidence_h: %f, confidence_y: %f, time stamp: %f_%f_%f_%f_%f');
stage_pos=[f{2} f{3}];
stage_pos=stage_pos-stage_pos(1,:);
stage_pos=[stage_pos(:,1)/10000 stage_pos(:,2)/12800];

stage_pos_1(:,1) = interp1(1:length(stage_pos),stage_pos(:,1),1:length(stage_pos)/length(blueOn5):length(stage_pos));
stage_pos_1(:,2) = interp1(1:length(stage_pos),stage_pos(:,2),1:length(stage_pos)/length(blueOn5):length(stage_pos));

stage_dist=zeros(length(stage_pos_1),1);
for i=1:length(stage_dist)
    stage_dist(i)=norm(stage_pos_1(1,:)-stage_pos_1(i,:),2);
end

% bout检测
bouts = boutDetect(stage_pos_1);

% 统计bout数量
% blueOn=1;laserOn=0;
% blueOn=1;laserOn=1;
% blueOn=0;laserOn=0;
% 黄光的起始帧和结束帧
laser_start=find(diff(laserOn5)==1);
laser_end=find(diff(laserOn5)==-1);
% 蓝光的起始帧和结束帧
blue_start=find(diff(blueOn5)==1);
blue_end=find(diff(blueOn5)==-1);

periodIdx=zeros(length(bouts.start),1);
stageIdx=zeros(length(bouts.start),1);
% 第一个周期
for i=1:length(bouts.start)
   if(bouts.start(i)<blue_end(1))
       periodIdx(i)=1;
   end
end
for i=1:length(bouts.start)   % 没有黄光和蓝光
   if(bouts.start(i)<blue_start(1))
       stageIdx(i)=1;
   end
end
for i=1:length(bouts.start)   % 蓝光
   if(bouts.start(i)>blue_start(1)&&bouts.start(i)<laser_start(1))
       stageIdx(i)=2;
   end
end
for i=1:length(bouts.start)   % 蓝光+黄光
   if(bouts.start(i)>laser_start(1)&&bouts.start(i)<laser_end(1))
       stageIdx(i)=3;
   end
end
for i=1:length(bouts.start)   % 蓝光
   if(bouts.start(i)>laser_end(1)&&bouts.start(i)<blue_end(1))
       stageIdx(i)=4;
   end
end

% 第二个周期
for i=1:length(bouts.start)
   if(bouts.start(i)>blue_end(1)&&bouts.start(i)<blue_end(2))
       periodIdx(i)=2;
   end
end
for i=1:length(bouts.start)   % 没有黄光和蓝光
   if(bouts.start(i)>blue_end(1)&&bouts.start(i)<blue_start(2))
       stageIdx(i)=1;
   end
end
for i=1:length(bouts.start)   % 蓝光
   if(bouts.start(i)>blue_start(2)&&bouts.start(i)<laser_start(2))
       stageIdx(i)=2;
   end
end
for i=1:length(bouts.start)   % 蓝光+黄光
   if(bouts.start(i)>laser_start(2)&&bouts.start(i)<laser_end(2))
       stageIdx(i)=3;
   end
end
for i=1:length(bouts.start)   % 蓝光
   if(bouts.start(i)>laser_end(2)&&bouts.start(i)<blue_end(2))
       stageIdx(i)=4;
   end
end

% 第三个周期
for i=1:length(bouts.start)
   if(bouts.start(i)>blue_end(2)&&bouts.start(i)<blue_end(3))
       periodIdx(i)=3;
   end
end
for i=1:length(bouts.start)   % 没有黄光和蓝光
   if(bouts.start(i)>blue_end(2)&&bouts.start(i)<blue_start(3))
       stageIdx(i)=1;
   end
end
for i=1:length(bouts.start)   % 蓝光
   if(bouts.start(i)>blue_start(3)&&bouts.start(i)<laser_start(3))
       stageIdx(i)=2;
   end
end
for i=1:length(bouts.start)   % 蓝光+黄光
   if(bouts.start(i)>laser_start(3)&&bouts.start(i)<laser_end(3))
       stageIdx(i)=3;
   end
end
for i=1:length(bouts.start)   % 蓝光
   if(bouts.start(i)>laser_end(3)&&bouts.start(i)<laser_start(4))
       stageIdx(i)=4;
   end
end
for i=1:length(bouts.start)   % 蓝光+黄光
   if(bouts.start(i)>laser_start(4)&&bouts.start(i)<laser_end(4))
       stageIdx(i)=5;
   end
end
% 画图 不同阶段的bout数量
bouts.idx=[periodIdx stageIdx];
id=[int2str(periodIdx) int2str(stageIdx)];
N=tabulate(id);
idx=[];
num=[];
for i=1:length(N)
    idx=cat(1,idx,N{i,1});
    num=cat(1,num,N{i,2});
end
bar(num)
set(gca,'xticklabel',idx,'box','off','xtick',1:length(idx))
ylabel('bout nums')
% 不同阶段bout的平均长度
id=str2num(id);
temp=zeros(size(N,1),1);
for i=1:length(idx)
    temp(i)=sum(bouts.len(find(id==str2num(idx(i,:)))))/length(find(id==str2num(idx(i,:))));
end
bar(temp)
set(gca,'xticklabel',idx,'box','off','xtick',1:length(idx))
ylabel('bout length')



%% 20221001_1011_g8s-lssm-tph2-chriR_11dpf   0.3mW弱蓝光常亮
clear
path='E:\online-opto-data\data-to-analysis\20221001_1011_g8s-lssm-tph2-chriR_11dpf\20221001_1011_g8s-lssm-tph2-chriR_11dpf.txt';
ExpLogs=readExpLogsFromTXT(path);
laserOn=ExpLogs.laserOn;
laserOn(end)=0;
headingAngle=ExpLogs.rotationAngleX;

blueOn=zeros(length(laserOn),1);
blueOn(3154:15566)=1;
blueOn(18702:end)=1;
blueOn(end)=0;

figure
plot(headingAngle);
hold on
drawShadow(blueOn,360,'b');
drawShadow(laserOn,360,'y');
xlabel('frames')
ylabel('headingAngle')
hold off

blueOn5 = repelem(blueOn,5);
laserOn5 = repelem(laserOn,5);

% 台子的移动距离
path='E:\online-opto-data\data-to-analysis\20221001_1011_g8s-lssm-tph2-chriR_11dpf\Stage_postion2022_10_1-10_14_15.txt';
fid=fopen(path);
f=textscan(fid,'%f, x: %f, y: %f, detection: %f, head: [%f, %f], yolk: [%f, %f], confidence_h: %f, confidence_y: %f, time stamp: %f_%f_%f_%f_%f');
stage_pos=[f{2} f{3}];
stage_pos=stage_pos-stage_pos(1,:);
stage_pos=[stage_pos(:,1)/10000 stage_pos(:,2)/12800];

stage_pos_1(:,1) = interp1(1:length(stage_pos),stage_pos(:,1),1:length(stage_pos)/length(blueOn5):length(stage_pos));
stage_pos_1(:,2) = interp1(1:length(stage_pos),stage_pos(:,2),1:length(stage_pos)/length(blueOn5):length(stage_pos));

stage_dist=zeros(length(stage_pos_1),1);
for i=1:length(stage_dist)
    stage_dist(i)=norm(stage_pos_1(1,:)-stage_pos_1(i,:),2);
end

% bout检测
bouts = boutDetect(stage_pos_1);
% 黄光的起始帧和结束帧
laser_start=find(diff(laserOn5)==1);
laser_end=find(diff(laserOn5)==-1);
% 蓝光的起始帧和结束帧
blue_start=find(diff(blueOn5)==1);
blue_end=find(diff(blueOn5)==-1);

periodIdx=zeros(length(bouts.start),1);
stageIdx=zeros(length(bouts.start),1);
% 第一个周期
for i=1:length(bouts.start)
   if(bouts.start(i)<blue_end(1))
       periodIdx(i)=1;
   end
end
for i=1:length(bouts.start)   % 没有黄光和蓝光
   if(bouts.start(i)<blue_start(1))
       stageIdx(i)=1;
   end
end
for i=1:length(bouts.start)   % 蓝光
   if(bouts.start(i)>blue_start(1)&&bouts.start(i)<laser_start(1))
       stageIdx(i)=2;
   end
end
for i=1:length(bouts.start)   % 蓝光+黄光
   if(bouts.start(i)>laser_start(1)&&bouts.start(i)<laser_end(1))
       stageIdx(i)=3;
   end
end
for i=1:length(bouts.start)   % 蓝光
   if(bouts.start(i)>laser_end(1)&&bouts.start(i)<laser_start(2))
       stageIdx(i)=4;
   end
end
for i=1:length(bouts.start)   % 蓝光+黄光
   if(bouts.start(i)>laser_start(2)&&bouts.start(i)<blue_end(1))
       stageIdx(i)=5;
   end
end

% 第二个周期
for i=1:length(bouts.start)
   if(bouts.start(i)>blue_end(1)&&bouts.start(i)<blue_end(2))
       periodIdx(i)=2;
   end
end
for i=1:length(bouts.start)   % 没有黄光和蓝光
   if(bouts.start(i)>blue_end(1)&&bouts.start(i)<blue_start(2))
       stageIdx(i)=1;
   end
end
for i=1:length(bouts.start)   % 蓝光
   if(bouts.start(i)>blue_start(2)&&bouts.start(i)<laser_start(3))
       stageIdx(i)=2;
   end
end
for i=1:length(bouts.start)   % 蓝光+黄光
   if(bouts.start(i)>laser_start(3)&&bouts.start(i)<laser_end(3))
       stageIdx(i)=3;
   end
end
for i=1:length(bouts.start)   % 蓝光
   if(bouts.start(i)>laser_end(3)&&bouts.start(i)<laser_start(4))
       stageIdx(i)=4;
   end
end
for i=1:length(bouts.start)   % 蓝光+黄光
   if(bouts.start(i)>laser_start(4)&&bouts.start(i)<blue_end(2))
       stageIdx(i)=5;
   end
end

% 画图 不同阶段的bout数量
bouts.idx=[periodIdx stageIdx];
id=[int2str(periodIdx) int2str(stageIdx)];
N=tabulate(id);
idx=[];
num=[];
for i=1:length(N)
    idx=cat(1,idx,N{i,1});
    num=cat(1,num,N{i,2});
end
bar(num)
set(gca,'xticklabel',idx,'box','off','xtick',1:length(idx))
ylabel('bout nums')
% 不同阶段bout的平均长度
id=str2num(id);
temp=zeros(size(N,1),1);
for i=1:length(idx)
    temp(i)=sum(bouts.len(find(id==str2num(idx(i,:)))))/length(find(id==str2num(idx(i,:))));
end
bar(temp)
set(gca,'xticklabel',idx,'box','off','xtick',1:length(idx))
ylabel('bout length')



%% 221008上午 第一个周期用0.3mW弱蓝光常亮，第二个周期用23mW强蓝光给1ms 闪烁模式
path='E:\online-opto-data\20221008_1033_g8s-lssm-tph2-chri_8dpf\20221008_1033_g8s-lssm-tph2-chri_8dpf.txt';
ExpLogs=readExpLogsFromTXT(path);
laserOn=ExpLogs.laserOn;
headingAngle=ExpLogs.rotationAngleX;

blueOn=zeros(length(laserOn),1);
blueOn(3471:17039)=1;
blueOn(20167:end)=1;

figure
plot(headingAngle);
hold on
drawShadow(blueOn,360,'b');
drawShadow(laserOn,360,'y');
xlabel('frames')
ylabel('headingAngle')


%% 221008下午  前两个周期0.3mW弱蓝光，第三个周期给3mW，10ms闪烁模式，第四个周期给1.5mW，20ms闪烁，第五个周期给0.6mW，50ms闪烁
path='D:\online-opto-data\20221008_1623_g8s-lssm-tph2-chri_8dpf\20221008_1623_g8s-lssm-tph2-chri_8dpf.txt';
ExpLogs=readExpLogsFromTXT(path);
laserOn=ExpLogs.laserOn;
headingAngle=ExpLogs.rotationAngleX;

blueOn=zeros(length(laserOn),1);
blueOn(3073:15598)=1;   %0.3mW 常亮蓝光
blueOn(18693:31368)=1;  %0.3mW 常亮蓝光
blueOn(34707:44023)=1;  % 3mW 10ms闪烁
blueOn(45120:52340)=1;  % 1.5mW 20ms闪烁
blueOn(53758:end)=1;    % 0.6mW 50ms闪烁

figure
plot(headingAngle);
hold on
drawShadow(blueOn,360,'b');
drawShadow(laserOn,360,'y');
xlabel('frames')
ylabel('headingAngle')


%% 20221009 上午 第一个周期0.3mW弱蓝光，第二个周期1.5mW蓝光强度，20ms 闪烁  agar破损鱼活动范围小
path='E:\online-opto-data\20221009_1038_g8s-lssm-tph2-chri_9dpf\20221009_1038_g8s-lssm-tph2-chri_9dpf.txt';
ExpLogs=readExpLogsFromTXT(path);
laserOn=ExpLogs.laserOn;
headingAngle=ExpLogs.rotationAngleX;

blueOn=zeros(length(laserOn),1);
blueOn(3029:15652)=1;   %0.3mW 常亮蓝光
blueOn(18757:end)=1;  %1.5mW 20ms闪烁蓝光  最后10min agar破损范围扩大，鱼的可活动范围非常小

figure
plot(headingAngle);
hold on
drawShadow(blueOn,360,'b');
drawShadow(laserOn,360,'y');
xlabel('frames')
ylabel('headingAngle')