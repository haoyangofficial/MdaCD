python general/clip_inference.py --src_path /home/haoyang/data/data/MdaCD/Dataset/SYSUCD \
                                 --split train val test \
                                 --img_split time1 time2 \
                                 --model_name ViT-B/16 \
                                 --class_names_path general/rscls.txt \
                                 --device cuda:0 \
                                 --tag 56_vit16

python general/clip_inference.py --src_path /home/haoyang/data/data/MdaCD/Dataset/LEVIRCD \
                                 --split train val test \
                                 --img_split time1 time2 \
                                 --model_name ViT-B/16 \
                                 --class_names_path general/rscls.txt \
                                 --device cuda:0 \
                                 --tag 56_vit16
