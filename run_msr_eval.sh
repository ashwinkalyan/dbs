for lambda_msr in 0 0.01 0.02 0.05 0.075 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
        echo "lambda_msr"
        echo $lambda_msr
        th eval.lua -gpuid 1 -model ../data/models/model_id_87500_7.t7 -image_folder ../data/coco_test/ -div_msr_model ../data/models/language_model.t7 -batch_size 1 -M 1 -B 20 -lambda_msr $lambda_msr -div_msr 1 -num_images -1 -append 'val' -dump_json 1 -verbose 1 2>&1 | tee log
done
