%% 对齐两个时间戳
path='E:\work\20221019_1609_g8s-lssm-chriR_11dpf\';
path50=[path,'Stage_postion2022_10_19-16_12_8.txt'];
path340=[path,'2022_10_19-16_5_0\time_stamps.txt'];
% path50='E:\work\20221017_1453_g8s-lssm-chriR_9dpf\Stage_postion2022_10_17-15_1_10.txt';
% path340='E:\work\20221017_1453_g8s-lssm-chriR_9dpf\2022_10_17-14_51_18\time_stamps.txt';

fid=fopen(path50);
f=textscan(fid,'%f, x: %f, y: %f, detection: %f, head: [%f, %f], yolk: [%f, %f], confidence_h: %f, confidence_y: %f, time stamp: %f_%f_%f_%f_%f');
timeStamp50=[f{11} f{12} f{13} f{14} f{15}];
ts50=caculateTimeStamp(timeStamp50);


fid=fopen(path340);
f2=textscan(fid,'%f_%f_%f_%f_%f');
timeStamp340=[f2{1} f2{2} f2{3} f2{4} f2{5}];
ts340=caculateTimeStamp(timeStamp340);

ts340_50=[ts340,zeros(length(ts340),1)];
c=1;
for i=1:length(ts50)
    for j=c:length(ts340)
        if(ts50(i)<ts340(j))
            ts340_50(j,2)=i;
            c=j;
            break;
        end
    end
end

save([path,'timestamp.mat'],'ts50','ts340','ts340_50','timeStamp340','timeStamp50');