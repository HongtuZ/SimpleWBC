
# bash eval_teacher.sh 0927_twist_teacher

export CUDA_VISIBLE_DEVICES=0

task_name="orca_priv_mimic"
proj_name="orca_priv_mimic"
exptid=$1

cd legged_gym/legged_gym/scripts

# Run the eval script
python play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 1 \
                --record_video \
