path='E:\online-opto-data\20221116_1627_g8s-lssm-huc-none_8dpf\';
load([path,'boutInfo.mat'])


bout_leftLaser_len=bouts.len(find(bouts.leftLaser==1));
bout_rightLaser_len=bouts.len(find(bouts.rightLaser==1));
bout_noLaser_len=bouts.len(find(bouts.noLaser==1));



polarscatter(-deg2rad(bout_leftLaser_sumAngle),bout_leftLaser_len)
pax = gca;
pax.ThetaDir = 'clockwise';		   % 按顺时针方式递增
pax.ThetaZeroLocation = 'top';     % 将0度放在顶部  
% pax.ThetaAxisUnits = 'radians';  %角度显示为pi
hold on
polarscatter(-deg2rad(bout_rightLaser_sumAngle),bout_rightLaser_len)
polarscatter(-deg2rad(bout_noLaser_sumAngle),bout_noLaser_len)
legend('Left Laser','Right Laser','No Laser')
savefig([path,'Bout-statistics.fig'])
hold off

%% left laser
bout_leftLaser_pos=bouts.pos(find(bouts.leftLaser==1));
bout_leftLaser_angle=bouts.angle(find(bouts.leftLaser==1));
bout_leftAngleLen=cell(length(bout_leftLaser_pos),1);
for i=1:length(bout_leftLaser_pos)
    pos=bout_leftLaser_pos{i};
    d=zeros(length(pos),1);
    
    for j=1:length(pos)
        d(j)=norm(pos(1,:)-pos(j,:),2);
    end
    bout_leftAngleLen{i}=d;
end

for i=1:length(bout_leftLaser_angle)
    ang=-deg2rad(smooth(bout_leftLaser_angle{i}(1:length(bout_leftAngleLen{i}))));
    dis=bout_leftAngleLen{i};
%     dis=smooth(dis);
    polarplot(ang,dis)
%     idx(i)
    hold on
%     pause
end
pax = gca;
pax.ThetaDir = 'clockwise';		   % 按顺时针方式递增
pax.ThetaZeroLocation = 'top';     % 将0度放在顶部  
% pax.ThetaAxisUnits = 'radians';  %角度显示为pi
savefig([path,'Bout-leftLaser-Path.fig'])
hold off


%% right laser
bout_rightLaser_pos=bouts.pos(find(bouts.rightLaser==1));
bout_rightLaser_angle=bouts.angle(find(bouts.rightLaser==1));
bout_rightAngleLen=cell(length(bout_rightLaser_pos),1);
for i=1:length(bout_rightLaser_pos)
    pos=bout_rightLaser_pos{i};
    d=zeros(length(pos),1);
    
    for j=1:length(pos)
        d(j)=norm(pos(1,:)-pos(j,:),2);
    end
    bout_rightAngleLen{i}=d;
end

for i=1:length(bout_rightLaser_angle)
polarplot(-deg2rad(smooth((bout_rightLaser_angle{i}(1:length(bout_rightAngleLen{i}))))),bout_rightAngleLen{i})
hold on
end
pax = gca;
pax.ThetaDir = 'clockwise';		   % 按顺时针方式递增
pax.ThetaZeroLocation = 'top';     % 将0度放在顶部  
% pax.ThetaAxisUnits = 'radians';  %角度显示为pi
savefig([path,'Bout-rightLaser-Path.fig'])
hold off

%% no laser
bout_noLaser_pos=bouts.pos(find(bouts.noLaser==1));
bout_noLaser_angle=bouts.angle(find(bouts.noLaser==1));
bout_noAngleLen=cell(length(bout_noLaser_pos),1);
for i=1:length(bout_noLaser_pos)
    pos=bout_noLaser_pos{i};
    d=zeros(length(pos),1);
    
    for j=1:length(pos)
        d(j)=norm(pos(1,:)-pos(j,:),2);
    end
    bout_noAngleLen{i}=d;
end

for i=1:length(bout_noLaser_angle)
polarplot(-deg2rad(smooth(bout_noLaser_angle{i}(1:length(bout_noAngleLen{i})))),bout_noAngleLen{i},'linewidth',1)
hold on
end
pax = gca;
pax.ThetaDir = 'clockwise';		   % 按顺时针方式递增
pax.ThetaZeroLocation = 'top';     % 将0度放在顶部  
% pax.ThetaAxisUnits = 'radians';  %角度显示为pi
savefig([path,'Bout-noLaser-Path.fig'])
hold off