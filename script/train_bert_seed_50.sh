
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

run_name="UniArks"
model_name="bert-large-cased"
pt_lr="1e-5"
ad_lr="1e-5"
use_original_template=true
use_adaptor=true
aug_loss=true


echo "***** hyperparameters *****"
echo "run_name: $run_name"
echo "model_name: $model_name"
echo "learning_rate: prompt_encoder: $pt_lr, adapter: $ad_lr"
echo "loss_type: $loss_type, para_loss $para_loss, aug_loss $aug_loss"
echo "load_prompt_checkpoint_dir: $load_prompt_checkpoint_dir"
echo "use_adaptor: $use_adaptor, use_original_template:$use_original_template, use_lm_finetune:$use_lm_finetune"
echo "******************************"

python -u cli.py \
       --run_name ${run_name}\
       --model_name ${model_name}\
       --pt_lr ${pt_lr}\
       --ad_lr ${ad_lr}\
       --use_original_template ${use_original_template}\
       --use_adaptor ${use_adaptor}\
       --aug_loss ${aug_loss}\


echo ""
echo "============"
echo "job finished successfully"sss
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"