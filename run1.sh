if nvcc -o main1 -Xcompiler -fopenmp main1.cu;
then
	echo `date`
	for SIZE in 8 800 80000 8000000 800000000
	do
		echo 
		echo "--- Data size: $SIZE ---"
		echo "CPU:"
		CUDA_VISIBLE_DEVICES=0 ./main1 -ck $SIZE
		echo "GPU:"
		CUDA_VISIBLE_DEVICES=0 ./main1 -gk $SIZE
	done
fi
