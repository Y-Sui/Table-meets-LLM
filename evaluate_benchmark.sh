# form benchmark evaluation
#python evaluate_benchmark.py --task_group form --log_file_path form_benchmarks_20230115_log --model text003 text002
#===================
# table benchmark evaluation
#python evaluate_benchmark.py --task_group table --log_file_path table_benchmarks_20230116_log --model text003 text002

## 20230118_zero-shot
#python evaluate_benchmark.py --task_group table --log_file_path table_benchmarks_20230118_zero_shot_log --linearize_list html_0_1_3 --model text003 text002 text001
#
## 20230119_1-shot (only html)
#python evaluate_benchmark.py --task_group table --log_file_path table_benchmarks_20230119_1_shot_log --model text003 text002 --linearize_list html_0_1_3 html_0_1 html_1_3 html_0_3 html_1

## 20230119_1-shot
#python evaluate_benchmark.py --task_group table --log_file_path table_benchmarks_20230119_1_shot_log --model text003 text002

# 20230120_change-order
#python evaluate_benchmark.py --task_group table --log_file_path table_benchmarks_20230120_change_order_log --model text003

# 20230120_zero-shot
python evaluate_benchmark.py --task_group table --log_file_path table_benchmarks_20230121_zero_shot_v2 --model text003
