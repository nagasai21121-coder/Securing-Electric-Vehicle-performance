
from flask import Flask, render_template,request,session
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('agg')
import sqlite3



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/adminlogin')
def AdminLogin():
    return render_template('AdminApp/AdminLogin.html')

@app.route('/AdminAction', methods=['POST'])
def AdminAction():
    if request.method == 'POST':
        username=request.form['username']
        password=request.form['password']

        if username=='Admin' and password=='Admin':
            return render_template("AdminApp/AdminHome.html")
        else:
            context={'msg':'Login Failed..!!'}
            return render_template("AdminApp/AdminLogin.html",**context)

@app.route('/AdminHome')
def AdminHome():
    return render_template("AdminApp/AdminHome.html")

@app.route('/Upload')
def Upload():
    return render_template("AdminApp/Upload.html")



UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

global dataset,filepath
@app.route('/UploadAction', methods=['POST'])
def UploadAction():
    global dataset,filepath
    if 'dataset' not in request.files:
        return "No file part"
    file = request.files['dataset']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    dataset = pd.read_csv(filepath)
    columns = dataset.columns.tolist()
    rows = dataset.head().values.tolist()
    return render_template('AdminApp/ViewDataset.html', columns=columns, rows=rows)

global dataset, X_train, X_test, y_train, y_test
@app.route('/preprocess')
def preprocess():
    global dataset, X_train, X_test, y_train, y_test

    dataset = pd.read_csv("Dataset/ev_fault_dataset.csv")
    dataset.dropna(inplace=True)

    dataset['Fault_Type']=dataset['Fault_Type'].map({'Healthy':0, 'Double-Line Fault':1, 'Three-Phase Fault':2})

    # Split data
    X = dataset.drop(columns=['Fault_Type'])
    y = dataset['Fault_Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return render_template('AdminApp/SplitStatus.html', total=len(X), train=len(X_train),test=len(X_test))


global results
@app.route('/runAlgorithms')
def runAlgorithms():
    global X_train,X_test,results
    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save Scaler
    joblib.dump(scaler, "Model/scaler.pkl")

    # Models
    models = {
        'Decision_Tree': DecisionTreeClassifier(),
        'Logistic_Regression': LogisticRegression(max_iter=1000),
        'SGD': SGDClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'XGBoost': GradientBoostingClassifier(),
        'KNN': KNeighborsClassifier()
    }

    # Train, Save, and Evaluate Models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"Model/{name}.pkl")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

    # Voting Classifier
    voting_clf = VotingClassifier(estimators=[
        ('DT', DecisionTreeClassifier()),
        ('LR', LogisticRegression(max_iter=1000)),
        ('SGD', SGDClassifier()),
        ('AB', AdaBoostClassifier()),
        ('XGB', GradientBoostingClassifier()),
        ('KNN', KNeighborsClassifier())
    ], voting='hard')

    voting_clf.fit(X_train, y_train)
    joblib.dump(voting_clf, "Model/Voting_Classifier.pkl")
    y_pred_voting = voting_clf.predict(X_test)
    voting_acc = accuracy_score(y_test, y_pred_voting)
    results['Voting Classifier'] = voting_acc
    print(f"Voting Classifier Accuracy: {voting_acc:.4f}")
    print(classification_report(y_test, y_pred_voting))

    # Plot Accuracy Comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Machine Learning Models')
    plt.xticks(rotation=30)
    plt.savefig('static/model_accuracy.png')
    plt.close()

    # Save results to HTML table
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    html_table = results_df.to_html(index=False)
    html_page = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Accuracy</title>
        <style>
            table {{ width: 50%; border-collapse: collapse; margin: 20px auto; }}
            th, td {{ border: 1px solid black; padding: 10px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h2 style="text-align:center;">Machine Learning Model Accuracy</h2>
        <img src="/static/model_accuracy.png" alt="Model Accuracy Graph" style="display: block; margin: 20px auto; width: 60%;">
        {html_table}
    </body>
    </html>
    """
    with open("templates/results.html", "w") as f:
        f.write(html_page)


    return render_template('results.html', msg="All Model Generated Successfully..!!")



@app.route('/comparison')
def comparison():
    global results
    # Plot comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Machine Learning Models')
    plt.xticks(rotation=30)
    plt.show()
    plt.savefig('Static/model_accuracy.png')
    plt.close()
    return render_template('AdminApp/Grpah.html')

@app.route('/userlogin')
def userlogin():
    return render_template('UserApp/Login.html')

@app.route('/register')
def register():
    return render_template('UserApp/Register.html')
    
@app.route('/RegAction', methods=['POST'])
def RegAction():
    if request.method == 'POST':
        name=request.form['name']
        email=request.form['email']
        mobile=request.form['mobile']
        username=request.form['username']
        password=request.form['password']

        con=sqlite3.connect('database.db')
        cur=con.cursor()
        #cur.execute("create table user(name varchar(100),email varchar(200),mobile varchar(200),username varchar(100),password varchar(100))")
        cur.execute("select * from user where username='"+username+"'and password='"+password+"'")
        data=cur.fetchone()
        if data is None:
            cur=con.cursor()
            cur.execute("insert into user values('"+name+"','"+email+"','"+mobile+"','"+username+"','"+password+"')")
            con.commit()
            return render_template('UserApp/Register.html', msg="Successfully Registered..!!")
        else:
            return render_template('UserApp/Register.html', msg="username and password is already exist..!!")

app.secret_key = '123'
@app.route('/UserAction', methods=['POST'])
def UserAction():
    username=request.form['username']
    password=request.form['password']

    con=sqlite3.connect('database.db')
    cur=con.cursor()
    cur.execute("select * from user where username='"+username+"'and password='"+password+"'")
    data=cur.fetchone()
    if data is None:
        return render_template('UserApp/Login.html', msg="Login Failed..!!")
    else:
        session['username'] =data[3]
        return render_template('UserApp/Home.html',username=session['username'])

@app.route('/Detect')
def Detect():
    return render_template('UserApp/Detect.html')

@app.route('/DetectAction', methods=['POST'])
def DetectAction():

    current = float(request.form['current'])
    voltage = float(request.form['voltage'])
    output_speed = float(request.form['Output_Speed'])
    m_speed = float(request.form['m_speed'])
    hall_sensor = float(request.form['hall_sensor'])




    input_data = np.array([[current, voltage, output_speed, m_speed, hall_sensor]])

    test_model = joblib.load("Model/Decision_Trees.pkl")

    prediction = test_model.predict(input_data)[0]
    output=""
    if prediction==0:
        output="Healthy"
    elif prediction ==1:
        output = "Double-Line Fault"
    else:
        output = "Three-Phase Fault"


    return render_template('UserApp/Result.html', result=output)

  # dataset['Fault_Type']=dataset['Fault_Type'].map({'Healthy':0, 'Double-Line Fault':1, 'Three-Phase Fault':2})
if __name__ == '__main__':
    app.run(debug=True)


