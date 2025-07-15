#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
log_dir="logs_${now}"

mkdir -p "$log_dir"

commands=(
"python -u /home/dell/sx/sgformer/SGFormer/large/new/main_graph.py --method AGGT --dataset molhiv --metric rocauc --lr 0.0001 --hidden_channels 256 --use_graph --graph_weight 0.5 --gnn_num_layers 3 --gnn_dropout 0.3 --gnn_weight_decay 0.0 --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act --gnn_for_origin_num_layers 4 --trans_num_layers 2 --trans_dropout 0.3 --trans_weight_decay 0.0 --trans_use_residual --trans_use_weight --trans_use_bn --n_hypers 4 --hyper_weight_decay 0.0 --hyper_dropout 0.3 --hyper_diverse_rate 1 --hyper_weight 0.5 --seed 123 --runs 1 --epochs 500 --eval_step 1 --device 3 --batch_size 64"
)

pids=()

for i in "${!commands[@]}"; do
    outfile="${log_dir}/run_output_job${i}.txt"
    echo "Running job $i, outputting to $outfile"
    
    {
        echo "=== Job $i Started ==="
        echo "Command: ${commands[$i]}"
        bash -c "${commands[$i]}"
        echo "=== Job $i Finished ==="
    } >> "$outfile" 2>&1 &

    pids+=($!)
done

pidfile="${log_dir}/pids.txt"
echo "${pids[@]}" > "$pidfile"

echo "Started all jobs! PIDs saved to $pidfile"
