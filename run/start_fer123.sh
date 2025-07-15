#!/bin/bash

now=$(date +"%Y%m%d_%H%M%S")
log_dir="logs_${now}"

mkdir -p "$log_dir"

commands=(
"python -u /home/dell/sx/sgformer/SGFormer/large/new/main_fer123happyandsad.py --method AGGT --dataset fer123happyandsad --metric rocauc --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
--gnn_num_layers 4 --gnn_dropout 0.5 --gnn_weight_decay 0.0 --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act --gnn_for_origin_num_layers 4 --gnn_for_origin_weight_decay 0.0 \
--trans_num_layers 2 --trans_dropout 0.5 --trans_weight_decay 0.0 --trans_use_residual --trans_use_weight --trans_use_bn \
--n_hypers 8 --hyper_weight_decay 0.0 --hyper_dropout 0.5 --hyper_diverse_rate 0.001 --hyper_weight 0.5 \
--seed 123 --runs 5 --epochs 1000 --eval_step 10 --device 2 --batch_size 64"

"python -u /home/dell/sx/sgformer/SGFormer/large/new/main_fer123angryandfear.py --method AGGT --dataset fer123angryandfear --metric rocauc --lr 0.001 --hidden_channels 256 --use_graph --graph_weight 0.5 \
--gnn_num_layers 4 --gnn_dropout 0.5 --gnn_weight_decay 0.0 --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act --gnn_for_origin_num_layers 4 --gnn_for_origin_weight_decay 0.0 \
--trans_num_layers 2 --trans_dropout 0.5 --trans_weight_decay 0.0 --trans_use_residual --trans_use_weight --trans_use_bn \
--n_hypers 8 --hyper_weight_decay 0.0 --hyper_dropout 0.5 --hyper_diverse_rate 0.001 --hyper_weight 0.5 \
--seed 123 --runs 5 --epochs 1000 --eval_step 10 --device 2 --batch_size 64"
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
