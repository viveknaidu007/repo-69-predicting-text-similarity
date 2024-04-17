from flask import Flask, request, jsonify, render_template
import joblib
import spacy

# now we are loading spaCy model
nlp = spacy.load('en_core_web_lg')

# loading the trained model
model = joblib.load('semantic_similarity_model.pkl')

app = Flask(__name__)

# function to preprocess the text data
def preprocess_text(text):
    return nlp(text)

# function to calculate similarity score
def predict_similarity(text1, text2):
    # preprocessing text
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    
    # calculate similarity score using the model we trained
    similarity_score = model.predict([[text1.similarity(text2)]])[0]
    similarity_score = float(similarity_score)    
    return similarity_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # getting data from request
    data = request.get_json()
    text1 = data['text1']
    text2 = data['text2']
    
    # predicting similarity score
    similarity_score = predict_similarity(text1, text2)
    
    # format response
    response = {'similarity_score': similarity_score}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)    #add host =0.0.0.0 , port =5000 , if we are deploying aws 
