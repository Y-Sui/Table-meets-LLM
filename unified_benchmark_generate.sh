## zero-shot
python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective zero-shot --split validation --linearize_list html --use_partition_mark --use_format_explanation --use_role_prompting
# all format without 1-shot
python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective zero-shot --split validation --use_partition_mark --use_format_explanation --use_role_prompting --linearize_list html markdown xml json nl_sep
#=============================
## few-shot
python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective few-shot --split validation --linearize_list html --use_partition_mark --use_format_explanation --use_role_prompting
##=============================
## 1-shot, 0_1_3
python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective 1-shot --split validation --use_format_explanation --use_partition_mark --use_role_prompting --linearize_list html markdown xml json nl_sep
## w/o role_prompting 0_1
python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective 1-shot --split validation --use_format_explanation --use_partition_mark --linearize_list html markdown xml json nl_sep
## w/o partition_mark 1_3
python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective 1-shot --split validation --use_format_explanation --use_role_prompting --linearize_list html markdown xml json nl_sep
## w/o format_explanation 0_3
python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective 1-shot --split validation --use_role_prompting --use_partition_mark --linearize_list html markdown xml json nl_sep