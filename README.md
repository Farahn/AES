# AES
Automatic Essay Scoring

Neural models for essay scoring for TOEFL essays (https://catalog.ldc.upenn.edu/LDC2014T06) and ASAP essays (http://www.kaggle.com/c/asap-aes).

The models are pretrained using a discourse marker prediction task, natural language inference task, or using pretrained text representation from BERT (https://arxiv.org/pdf/1810.04805.pdf) or USE (https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46808.pdf). 

Details can be found in the paper “Automated Essay Scoring with Discourse Aware Neural Models” F. Nadeem, H. Nguyen, Y. Liu and M. Ostendorf, Proceedings of the 14th Workshop on Innovative Use of NLP for Building Educational Applications at ACL 2019.
Models can be downloaded at https://sites.google.com/site/nadeemf0755/research/automatic-essay-scoring

For the two data sets, ASAP and TOEFL (LDC), the first step is to run the data parse scripts, either ASAP_dataparse.ipynb or TOEFL_dataparse.ipynb. After that the training or testing scripts can be run for all models except the ones that use BERT. For the models using BERT, the BERT preprocessing scripts should be run before the training or testing (BERT_text_representation.ipynb). 
