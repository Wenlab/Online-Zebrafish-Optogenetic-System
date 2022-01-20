
function Am_1=read3DAffineMatrixFromITK(filename)
    fileID = fopen(filename);
    C = textscan(fileID,'%s %f %f %f','headerlines', 6);
    fclose(fileID);
    Am=[C{2},C{3},C{4}];
    Am=Am(1:4,:);
    Am_1=reshape(Am',[1,12]);
end