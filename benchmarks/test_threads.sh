torch_threads=()

# Loop similar to 'for i in range(9)' in Python
for ((i=0; i<9; i++)); do
    # Compute the value and add to the array
    torch_threads+=($((4*i + 8)))
done

cpp_threads=()

for ((i=0; i<9; i++)); do
    cpp_threads+=($((4*i + 24)))
done

for torch_thread in "${torch_threads[@]}"; do
    for cpp_thread in "${cpp_threads[@]}"; do
        echo "Running with torch_threads=$torch_thread and cpp_threads=$cpp_thread"
        python3 latency.py --torch_threads $torch_thread --cpp_threads $cpp_thread
    done
done