% vector1: target-head
% vector2: head-headbacktip
% output: angle degree
function [theta]=two_vector_angle(vector1,vector2)
%     temp = angle(vector1(2)-sqrt(-1)*vector1(1));
%     theta = temp-angle(vector2(2)-sqrt(-1)*vector2(1));
%     theta = rad2deg(theta);

% cosin=dot(vector1,vector2)/(norm(vector1)*norm(vector2));
% theta=acos(cosin);
% theta=rad2deg(theta);
% cross=vector1(1)*vector2(2)-vector1(2)*vector2(1);
% if(cross<0)
%     theta=-theta;
% end

x1 = vector1(1);
y1 = vector1(2);
x2 = vector2(1);
y2 = vector2(2);
theta = atan2d(y1,x1) - atan2d(y2,x2);

if abs(theta) > 180
    theta = theta - 360*sign(theta);
end

end