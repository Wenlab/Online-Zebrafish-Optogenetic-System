#!/bin/bash
export CMTK_WRITE_UNCOMPRESSED=1


refName="/home/data2/backup_customer/kexin/fishData/210924/Obj_ref3.nii"
movingPath="/home/data2/backup_customer/kexin/fishData/210924/TM/"
savePath="/home/data2/backup_customer/kexin/fishData/210924/CMTKaffineResult/"
if [ ! -d "$savePath" ];then
    mkdir $savePath
fi

for file in $(ls ${movingPath})
do 
    if [ -d ${savePath}affine1st${file%%.*}.xform ];then
        continue
    fi

    if [ "${file##*.}" = "nii" ]; then
        start_time=$(date +%s)
        # make intiital transform
	    cmtk make_initial_affine --principal-axes ${refName} ${movingPath}${file} ${savePath}initial${file%%.*}.xform
        # affine registration, dofs = 12 (9, 6 as optional)
	    cmtk registration --initial ${savePath}initial${file%%.*}.xform --dofs 9,12 --exploration 8 --accuracy 0.05 --cr -o ${savePath}affine1st${file%%.*}.xform ${refName} ${movingPath}${file}

        # apply the transformation on the original 3-D stack	
	    cmtk reformatx -o ${savePath}Obj_1stAffined_${file%%.*}.nii --floating ${movingPath}${file} ${refName} ${savePath}affine1st${file%%.*}.xform


        end_time=$(date +%s)
		
		cost_time=$[ $end_time-$start_time ]

		# print out the i-th exec. time
		echo $num
		echo "Reg & Warp time is $(($cost_time/60))min $(($cost_time%60))s"
        echo "${movingPath}${file} regis done"

    fi
done