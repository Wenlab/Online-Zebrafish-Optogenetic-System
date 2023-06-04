function [] = saveTMparams(dir,rotationAngleXY,rotationAngleYZ,cropPoint,rescale)
%save x,y in dir
% so I can save in parfor loop
writematrix(rotationAngleXY,[dir,'rotationAngleXY.txt'])
writematrix(rotationAngleYZ,[dir,'rotationAngleYZ.txt'])
writematrix(cropPoint,[dir,'cropPoint.txt'])
writematrix(rescale,[dir,'rescale.txt'])

end