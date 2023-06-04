export CMTK_WRITE_UNCOMPRESSED=1

path="/home/data2/backup_customer/kexin/fishData/210824/CMTKaffineResult/"

dir=$(ls -l ${path} |awk '/^d/ {print $NF}')

for i in $dir
do 
    var=$(cmtk dof2mat ${path}$i)
    echo "$var" >>${path}CMTKaffineResult.txt
done