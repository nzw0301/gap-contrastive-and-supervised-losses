cd ../../

CONFIG="loss.normalize=true"

parallel -j 8 "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python toy_contrastive.py loss.neg_size={} ${CONFIG} seed={#} gpu_id={#}" ::: $(perl -e 'print "1 " x 8')
parallel -j 8 "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python toy_contrastive.py loss.neg_size={} ${CONFIG} seed={#} gpu_id={#}" ::: $(perl -e 'print "4 " x 8')
parallel -j 8 "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python toy_contrastive.py loss.neg_size={} ${CONFIG} seed={#} gpu_id={#}" ::: $(perl -e 'print "16 " x 8')
parallel -j 8 "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python toy_contrastive.py loss.neg_size={} ${CONFIG} seed={#} gpu_id={#}" ::: $(perl -e 'print "64 " x 8')
parallel -j 8 "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python toy_contrastive.py loss.neg_size={} ${CONFIG} seed={#} gpu_id={#}" ::: $(perl -e 'print "256 " x 8')
