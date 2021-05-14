from flask import Flask, render_template, request
import joblib
#from werkzeug import secure_filename
#import werkzeug

app=Flask(__name__)

#Loading model
vect=joblib.load('NLP_netfix_vectorizer.pkl')
clf=joblib.load('NLP_netflix_svm_model.pkl')


@app.route('/') #python decorator
def index(): #function associated with Route /
    #return "hello world"
    return render_template('sentiment_file.html')

# @app.route('/predict',methods=['POST'])
# def predict(x):
#     '''
#     For rendering results on HTML GUI
#     '''

#     data = [x for x in request.form.values()]

#     #connect my databse
    
#     #final_features = [np.array(int_features)]
#     #final_features = features
#    vector=vect.transform(data)
#    prediction = clf.predict(vector)
    
    # if prediction[0]==1.0:
    #     output="Positive"
    # else:
    #     output="Negative"
    
    # return render_template('sentiment_file.html', prediction_text='It is a {} sentiment'.format(output))

@app.route('/getfile', methods=['GET','POST'])
def getfile():
    if request.method == 'POST':

        # for secure filenames. Read the documentation.
        file = request.files['testfile']
        #filename = secure_filename(file.filename) 
        
        result = file.read()
        # os.path.join is used so that paths work in every operating system
        #file.save(os.path.join("wherever","you","want",filename))

        # You should use os.path.join here too.
        #with open("wherever/you/want/filename") as f:
            #file_content = f.read()
    
    else:
        result = request.args.get['testfile']

    data = [result]  
    vector=vect.transform(data)
    prediction = clf.predict(vector)
    if prediction[0]==1.0:
        output="Positive"
    else:
        output="Negative"
    
    return render_template('sentiment_file.html', prediction_text='It is a {} sentiment'.format(output))

        

if __name__ == "__main__": 
    app.run(debug=True)