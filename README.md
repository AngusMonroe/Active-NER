# Active-NLP

Partly refer [Active-NLP](https://github.com/asiddhant/Active-NLP)

For example to run a CNN_BiLSTM_CRF model on Conll dataset on full dataset, you can run

     $ python train_ner.py --usemodel CNN_BiLSTM_CRF --dataset conll

and to run active learning for CNN_BiLSTM_CRF model on  Conll dataset with "MNLP" acquisition function, you can run

     $ python active_ner.py --usemodel CNN_BiLSTM_CRF --dataset conll --acquiremethod mnlp

For plots to evaluate the result, you can run

     $ python eval.py
