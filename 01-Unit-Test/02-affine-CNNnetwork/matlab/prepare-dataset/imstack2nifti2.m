function imstack=imstack2nifti2(niftiImg)
    niftiImg = flip(niftiImg, 2);
    imstack=rot90(niftiImg,1);

end