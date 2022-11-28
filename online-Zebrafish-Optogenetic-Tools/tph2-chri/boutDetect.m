% 输入  台子坐标(x,y)

function bouts = boutDetect(stage_pos)

    stage_dist=zeros(length(stage_pos),1);
    for i=1:length(stage_dist)
        stage_dist(i)=norm(stage_pos(1,:)-stage_pos(i,:),2);
    end
    stage_dist(1:10)=0;stage_dist(end)=0;
    stage_dist_diff=diff(stage_dist);
    stage_dist_diff_abs=abs(stage_dist_diff);
%     figure
%     plot(stage_dist_diff_abs)
    bout_detect=zeros(length(stage_dist_diff_abs),1);
    bout_detect(find(stage_dist_diff_abs>0.02))=1;  % 台子位移>0.02mm认为发生了bout
    se = strel('disk',3);
    bout_detect_close=imclose(bout_detect,se);
    bout_detect_open=imopen(bout_detect_close,se);
    bout_detect_open(end)=0;
    bout_start=find(diff(bout_detect_open)==1)-3;   % 前后各延长一点
    bout_end=find(diff(bout_detect_open)==-1);
    
    bouts.start=bout_start;
    bouts.end=bout_end;
    
    bout_pos=cell(length(bout_start),1);
    for i=1:length(bout_pos)
        bout_pos{i}=stage_pos(bout_start(i):bout_end(i),:);
    end
    bouts.pos=bout_pos;
    
    len=zeros(length(bout_pos),1);
    for i=1:length(bout_pos)
        pos=bout_pos{i};
        d=0;
        for j=1:size(pos,1)-1
            d=d+norm(pos(j,:)-pos(j+1,:),2);
        end
        len(i)=d;
    end
    bouts.len=len;
end