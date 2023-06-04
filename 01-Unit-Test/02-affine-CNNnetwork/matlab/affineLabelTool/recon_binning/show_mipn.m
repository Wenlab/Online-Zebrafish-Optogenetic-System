function a=show_mipn(b)
a=[max(b,[],3) squeeze(max(b,[],2));squeeze(max(b,[],1))' zeros(size(b,3),size(b,3))];
figure;
imagesc(a);axis image;
end