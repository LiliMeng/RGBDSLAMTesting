%plot 3d data
function plot3DCameraPose(x,y,z)
figure
%test = xlsread('camera_pose.xlsx', 'A2:C1665');
test = xlsread('pose_after_optimization.xlsx', 'A1:C240');
x=test(:,1);
y=test(:,2);
z=test(:,3);

%plot3(x,y,z);
scatter3(x,y,z,'filled');
title('Camera Keyframe Poses After Optimization','FontSize', 20);
%scatter3(x,y,z,'filled')
xlabel('x position (meter)','FontSize', 20); % x-axis label
ylabel('y position (meter)','FontSize', 20);% y-axis label
zlabel('z position (meter)','FontSize', 20);% z-axis label

end

