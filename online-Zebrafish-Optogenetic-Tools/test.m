
x = pos';
A = [cosd(angle) -sind(angle);sin(angle) cosd(angle)];
y = A * x;
plot(x(1,:),x(2,:), 'b')
hold on
plot(y(1,:),y(2,:),'r')
% axis([-1 1.5 0 1.5])
grid on


x1 = [0;0];
x2 = [1;0];
x3 = [1;1];
x4 = [0;1];
x = [x1,x2,x3,x4,x1];
angle=-43.22;
angle=degtorad(angle);
% angle=40;
A = [cos(angle)   -sin(angle);sin(angle)   cos(angle)];
y = A * x;
plot(x(1,:),x(2,:), 'b')
hold on
plot(y(1,:),y(2,:),'r')
axis equal 
grid on
hold off