#!/bin/bash

# 定义命令列表
cmds=(
    'CUDA_VISIBLE_DEVICES="0" /home/zsl/.conda/envs/bz/bin/python /home/zsl/label_everything/label_everything.py --input_folder /home/zsl/label_everything/dataset4 --output_folder "/home/zsl/label_everything/dataset4结果" --input_text "fish" --score_threshold 0.05 --ape_version "D"'
    'CUDA_VISIBLE_DEVICES="0" /home/zsl/.conda/envs/bz/bin/python /home/zsl/label_everything/label_everything.py --input_folder /home/zsl/label_everything/dataset5 --output_folder "/home/zsl/label_everything/dataset5结果" --input_text "fish" --score_threshold 0.12 --ape_version "D"'
    'CUDA_VISIBLE_DEVICES="0" /home/zsl/.conda/envs/bz/bin/python /home/zsl/label_everything/label_everything.py --input_folder /home/zsl/label_everything/dataset6 --output_folder "/home/zsl/label_everything/dataset6结果" --input_text "fish" --score_threshold 0.12 --ape_version "D"'
    'CUDA_VISIBLE_DEVICES="0" /home/zsl/.conda/envs/bz/bin/python /home/zsl/label_everything/label_everything.py --input_folder /home/zsl/label_everything/dataset7 --output_folder "/home/zsl/label_everything/dataset7结果" --input_text "fish" --score_threshold 0.12 --ape_version "D"'
)

# 依次运行每个命令
for i in "${!cmds[@]}"; do
    echo "Running task $((i+1))..."
    eval "${cmds[i]}"  # 执行当前命令
    if [ $? -ne 0 ]; then
        echo "Task $((i+1)) failed. Stopping script."
        exit 1
    fi
    echo "Task $((i+1)) completed successfully."
done

echo "All tasks completed."
