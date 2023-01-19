# form benchmark evaluation
#python evaluate_benchmark.py --task_group form --log_file_path form_benchmarks_20230115_log --model text003 text002
#===================
# table benchmark evaluation
#python evaluate_benchmark.py --task_group table --log_file_path table_benchmarks_20230116_log --model text003 text002

# 20230118_zero-shot
#python evaluate_benchmark.py --task_group table --log_file_path table_benchmarks_20230118_zero_shot_log --linearize_list html_0_1_3 --model text003 text002 text001

# 20230116 1-shot
python evaluate_benchmark.py --task_group table --log_file_path table_benchmarks_20230116_log --linearize_list html_0_1 html_0 html_1 --model text003 text002 text001