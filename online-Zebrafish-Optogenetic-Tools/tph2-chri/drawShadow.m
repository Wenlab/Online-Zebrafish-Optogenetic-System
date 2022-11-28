function drawShadow(data,ver,color)
    data(end)=0;
    data_f=diff(data);
    data_start=find(data_f==1);
    data_end=find(data_f==-1);
    
    f=[1 2 3 4];
    for i=1:length(data_start)
        v=[data_start(i) 0;data_start(i) ver;data_end(i) ver;data_end(i),0];
        patch('Faces',f,'Vertices',v,'FaceColor',color,'edgealpha',0,'facealpha',0.2);
    end

end