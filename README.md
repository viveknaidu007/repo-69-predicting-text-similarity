# repo-69-predicting-text-similarity
here im building a model to find the text-similarity of sentences and experimenting their approaches and also deployment in aws

#we can see docuemntation for word embedding belowe resources
https://radimrehurek.com/gensim/models/word2vec.html

#to use pretrained models:
https://github.com/nishankmahore/word2vec-flask-api

#for huggingface model to use:
https://huggingface.co/fse/word2vec-google-news-300

#for references :
https://github.com/PradipNichite/Youtube-Tutorials/blob/main/GPT_3_Embeddings_Youtube_.ipynb

#form spacy models:
https://spacy.io/models/en   , #download en_core_web_lg to use
!python -m spacy download en_core_web_lg


#for installing requiremnts
pip install -r requirements.txt

#to run th ecode
python app.py




#for hosting the project on aws

upload files - .pkl , app.py , templates . requiremnts using scp command

using ssh client in aws connect to server using .pem key

install - 
sudo apt update
sudo apt upgrade
sudo apt install python3-pip

run the app.py 

and use nohup command to run continuely of our code in the entire internet , we can close our server terminal , it will run in aws server

after your work is done

kill the running file in server.
