img=niftiread('\\211.86.157.231\WenLabStorage2\Kexin\4bintest\210924\Obj_ref.nii');

MIPC=[max(img,[],3) squeeze(max(img,[],2));squeeze(max(img,[],1))' zeros(size(img,3),size(img,3))];
MIPC=uint16(MIPC);
figure(3)
imagesc(MIPC)

img = flip(img, 3);

niftiwrite(img, '\\211.86.157.231\WenLabStorage2\Kexin\4bintest\210924\Obj_ref2.nii');
% savenitfi(img,'\\211.86.157.231\WenLabStorage2\Kexin\4bintest\210824\Obj_ref2.nii');

