if nvcc -o main2 -Xcompiler -fopenmp main2.cu ;
then
    echo `date`
    for SIZE in 32 320 3200 32000
    do
        echo
        echo "--- Matrix size: $SIZE x $SIZE ---"
        echo "CPU:"
        ./main2 -cn $SIZE
        echo "GPU:"
        ./main2 -gn $SIZE
    done
fi
