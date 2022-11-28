function afterRotation=rotation3DFishImg(ObjRecon)
    BWObjRecon = ObjRecon>nanmean((nanmean(nanmean(ObjRecon)+4)));
    
    stats = regionprops3(BWObjRecon, 'Volume','Orientation');
    
    
    if(height(stats)>0)
    % 1st Rotation 
        prop = cell2mat(table2cell(stats));
        [maxv, index]=max(prop(:,1));
        RotateAngle= prop(index,2:4);
        afterRotation=imrotate(ObjRecon,-RotateAngle(1),'bicubic', 'crop');  %rotationһ��

        % filp image 
        ObjReconRedBW = afterRotation>nanmean((nanmean(nanmean(afterRotation)+4)));
        statsX = regionprops3(ObjReconRedBW,'volume','Centroid');
        propX = cell2mat(table2cell(statsX));
        [maxv, index]=max(propX(:,1));
        CentroID = propX(index,2:4);
        [X Y Z] = size(afterRotation);
        if CentroID(2) < (Y/2)
            afterRotation = imrotate(afterRotation, 180,'bicubic', 'crop');  %rotation����
        end
        
        
        % 2nd Rotation
        ObjReconRedBW = afterRotation>nanmean((nanmean(nanmean(afterRotation)+4)));
        statsX = regionprops3(ObjReconRedBW,'volume', 'Orientation');
        if(height(statsX)>0)
            propX = cell2mat(table2cell(statsX));
            [maxv, index]=max(propX(:,1));
            RotationAngle = propX(index,2:4);

            afterRotation=permute(afterRotation,[3 1 2]);
            afterRotation = imrotate(afterRotation,-RotationAngle(2),'bicubic', 'crop');  %rotation����
            afterRotation=permute(afterRotation,[2 3 1]);
%             afterRotation = imrotate(afterRotation,180,'bicubic', 'crop');  %rotation�Ĵ�
            
            
            MIPC=[max(afterRotation,[],3) squeeze(max(afterRotation,[],2));squeeze(max(afterRotation,[],1))' zeros(size(afterRotation,3),size(afterRotation,3))];
            MIPC=uint16(MIPC);
            %  figure(2); imagesc(MIPC);  
        end
    end
end