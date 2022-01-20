function savenitfi(imstack,filename)
    imstack = flip(imstack, 2);
    imstack=rot90(imstack,1);
    niftiwrite(imstack, filename);

end