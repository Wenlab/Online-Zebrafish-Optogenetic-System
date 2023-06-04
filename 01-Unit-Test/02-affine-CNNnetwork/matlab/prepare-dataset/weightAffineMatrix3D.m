function result=weightAffineMatrix3D(affineMatrices)
    affineMatrices(:,1)=(affineMatrices(:,1)-1)*10;
    affineMatrices(:,5)=(affineMatrices(:,5)-1)*10;
    affineMatrices(:,9)=(affineMatrices(:,9)-1)*10;
    affineMatrices(:,10)=affineMatrices(:,10)/77;
    affineMatrices(:,11)=affineMatrices(:,11)/95;
    affineMatrices(:,12)=affineMatrices(:,12)/52;
    result=affineMatrices*100;
end