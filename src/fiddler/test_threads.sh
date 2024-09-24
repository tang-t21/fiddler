torch_threads=()

# Loop similar to 'for i in range(9)' in Python
for ((i=0; i<9; i++)); do
    # Compute the value and add to the array
    torch_threads+=($((4*i + 8)))
done


for torch_thread in "${torch_threads[@]}"; do 
    echo "Running with torch_threads=$torch_thread"
    python3 infer.py --torch_threads $torch_thread --n-token=32 >> test_threads.txt
done