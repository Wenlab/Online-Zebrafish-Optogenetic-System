function [coordinates,himage] = neuronsSpatialDistribution(k,A,ObjRecon)
% Syntax: neuronsSpatialDistribution(k,A,ObjRecon)
% Input:
%         k: the indices of ROIs
%         A: footprint of ROIs(You can load the variable 'A3' from the segmentation results 'Coherence3.mat')
%         ObjRecon: pick one frame as background

% Output:
%         coordinates: the coordinates of selected neurons
%         himage: figure handle

% 
% Long description
%   Draw the spatial distribution of neurons. 

% July 21 updates:
% Add Line 75 for saving plots
% now dx = d1; dy = d2; dz = dz, no longer need shift.

    d1 = size(ObjRecon,1);
    d2 = size(ObjRecon,2);
    d3 = size(ObjRecon,3);
    dx = d1;
    dy = d2;
    dz = d3;
%     
    if (d1==dx) && (d2==dy) && (d3==dz)
        x_shift = 0;
        y_shift = 0;
        z_shift = 0;
    else
        x_shift = 140; % If the registered image is smaller than the original one, there should be a shift.
        y_shift = 0;
        z_shift = 60;
    end

    MIP = [max(ObjRecon,[],3) squeeze(max(ObjRecon,[],2));squeeze(max(ObjRecon,[],1))' zeros(size(ObjRecon,3),size(ObjRecon,3))];
    MIP = MIP/prctile(MIP(:),96);
    MIP = single(repmat(MIP,1,1,3)); % convert to 3 channels RGB image
    x_c = zeros(1,length(k));
    y_c = zeros(1,length(k));
    z_c = zeros(1,length(k));
    for index_k=1:length(k) % for each neuron, draw its contour
        temp = find(full(A(:,k(index_k))));
        [x,y,z] = ind2sub([dx,dy,dz],temp);
        x = x - x_shift;
        y = y - y_shift;
        z = z - z_shift;
        x_c(index_k) = round(mean(x));
		y_c(index_k) = round(mean(y));
		z_c(index_k) = round(mean(z));
        temp = zeros(size(MIP,1),size(MIP,2)); % the contour of the neuron on MIP image
        temp_index = sub2ind(size(MIP),x,y);
        temp(temp_index) = 1;
        temp_index = sub2ind(size(MIP),x,z+d2);
        temp(temp_index) = 1;
        temp_index = sub2ind(size(MIP),z+d1,y);
        temp(temp_index) = 1;
        temp = bwperim(temp);
        R = squeeze(MIP(:,:,1));
        G = squeeze(MIP(:,:,2));
        B = squeeze(MIP(:,:,3));
        color_k = squeeze(ind2rgb(mod(k(index_k),64),hsv));
        R(temp) = color_k(1);
        G(temp) = color_k(2);
        B(temp) = color_k(3);
        MIP(:,:,1) = R;
        MIP(:,:,2) = G;
        MIP(:,:,3) = B;
    end

    %% display
    himage=imshow(MIP);
    for index_k=1:length(k) % for each neuron, add a text
        color_k = squeeze(ind2rgb(mod(k(index_k),64),hsv));
        text(y_c(index_k),x_c(index_k),int2str(k(index_k)),'Color',color_k);
    end
    

 for i=1:length(k)
 coordinates{i}=[x_c(i),y_c(i),z_c(i)];
 end

end