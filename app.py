from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_mail import Mail, Message
import random
import string
from werkzeug.utils import secure_filename
import os
import plotly.express as px
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np,pandas as pd
import os
import csv
from dotenv import load_dotenv
import pdfkit
import pickle
from reportlab.pdfgen import canvas
from io import BytesIO
from flask.helpers import send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from io import BytesIO

app = Flask(__name__)
mail = Mail(app)
load_dotenv()

app.secret_key = 'MYSECRETKEY'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT') or 465)
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')
mail = Mail(app)
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(120), nullable=False)
  
    type_of_doctor = db.Column(db.String(50))

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    blood_group = db.Column(db.String(10), nullable=False)
    time_slot = db.Column(db.String(50), nullable=False)
    phone_number = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    type_of_doctor = db.Column(db.String(50))
    status = db.Column(db.String(20), default='Pending')
    prescription_file = db.Column(db.String(255))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('appointments', lazy=True))

def create_tables():
    with app.app_context():
        db.create_all()

def generate_random_string(length=10):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))

def send_mail(subject, recipient, body):
    msg = Message(subject, recipients=[recipient])
    msg.body = body
    mail.send(msg)
    
# Set the path to the directory containing text files
text_files_dir = os.path.join(os.path.dirname(__file__), 'static/prescriptions')

# Set the path to the directory where PDFs will be saved
pdf_output_dir = os.path.join(os.path.dirname(__file__), 'static/pdfs')

# Function to convert text file to PDF
def convert_to_pdf(file_path, output_path):
    with open(file_path, 'r') as file:
        content = file.read()

    pdfkit.from_string(content, output_path, {'title': 'PDF Conversion', 'footer-center': '[page]/[topage]'})
    
# ============================================================ model ============================================================ 


data = pd.read_csv(r"C:\Users\MSHigh School\Downloads\Remisense.ai-Webapp-main\Remisense.ai-Webapp-main\static\Data\Testing.csv")

df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)

indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]

dictionary = dict(zip(symptoms,indices))

def predict(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1, 1)).transpose()

    predicted_disease = dt.predict(user_input_label)[0]
    confidence_score = np.max(dt.predict_proba(user_input_label)) * 100  # Assuming decision tree has predict_proba method

    return predicted_disease, confidence_score

with open(r"C:\Users\MSHigh School\Downloads\Remisense.ai-Webapp-main\Remisense.ai-Webapp-main\static\Data\Testing.csv", newline='') as f:

        
        reader = csv.reader(f)
        symptoms = next(reader)
        symptoms = symptoms[:len(symptoms)-1]
        
# ============================================================ routes ============================================================ 

@app.route('/', methods=['GET', 'POST'])
def index():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        if user.type_of_doctor:
            appointments = Appointment.query.filter_by(type_of_doctor=user.type_of_doctor).all()
            return render_template('doctor-dashboard.html', username=username, appointments=appointments)
            
        else:
            user_appointments = user.appointments
            return render_template('patient-dashboard.html', username=username, user_appointments=user_appointments)
            
    return render_template('index.html')


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        Email = user.email 
        user_appointments = user.appointments
        return render_template('patient-profile.html', username=username,Email=Email, user_appointments=user_appointments)
    return render_template('index')

@app.route('/patient-register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        try:
            user = User(username=username, email=email, password=password)
            db.session.add(user)
            db.session.commit()
            session['user_id'] = user.id
            return redirect(url_for('index'))
        except IntegrityError:
            db.session.rollback()
            flash('Username already exists. Please choose a different username.', 'error')

    return render_template('patient-register.html')

@app.route('/doctor-register', methods=['GET', 'POST'])
def doctor_register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        type_of_doctor = request.form['type_of_doctor']
        user = User(username=username,email=email, password=password, type_of_doctor=type_of_doctor)
        db.session.add(user)
        db.session.commit()
        session['user_id'] = user.id
        return redirect(url_for('index'))
    return render_template('doctor-register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            flash('Wrong username or password. Please try again.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/book-appointment', methods=['GET', 'POST'])
def book_appointment():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = None
    
    user = User.query.get(session['user_id'])
    username = user.username
    
    # Fetch distinct types of doctors from the database
    doctor_types = db.session.query(User.type_of_doctor).distinct().all()
    doctor_types = [doctor[0] for doctor in doctor_types]

    if request.method == 'POST':
        name = request.form['name']
        age = int(request.form['age'])
        blood_group = request.form['blood_group']
        time_slot = request.form['time_slot']
        phone_number = request.form['phone_number']
        email = request.form['email']
        type_of_doctor = request.form['type_of_doctor']

        appointment = Appointment(
            name=name,
            age=age,
            blood_group=blood_group,
            time_slot=time_slot,
            phone_number=phone_number,
            email=email,
            type_of_doctor=type_of_doctor,
            user=user
        )

        db.session.add(appointment)
        db.session.commit()

        # Notify the doctor via email
        doctor_email = User.query.filter_by(type_of_doctor=type_of_doctor).first().email
        subject = 'New Appointment Request'
        body = f'Hello Doctor,\n\nYou have a new appointment request. Please log in to the system to approve or reject it.'
        send_mail(subject, doctor_email, body)

        return redirect(url_for('index'))

    return render_template('book-appointment.html',doctor_types=doctor_types,username=username)

@app.route('/approve-appointment/<int:appointment_id>')
def approve_appointment(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    doctor = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if appointment.type_of_doctor != doctor.type_of_doctor:
        return redirect(url_for('index'))

    appointment.status = 'Approved'
    db.session.commit()

    # Notify the patient via email
    subject = 'Appointment Approved'
    body = f'Hello {appointment.name},\n\nYour appointment has been approved. Please log in to the system to view the details.'
    send_mail(subject, appointment.email, body)

    return redirect(url_for('index'))

@app.route('/policy')
def policy():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('privacy-policy.html',username=username)
    return render_template('index.html')

@app.route('/Transforming_Healthcare')
def Transforming_Healthcare():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('blog_Transforming Healthcare.html',username=username)
    return render_template('index.html')

@app.route('/Holistic_Health')
def Holistic_Health():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('blog_Holistic Health.html',username=username)
    return render_template('index.html')

@app.route('/Nourishing_Body')
def Nourishing_Body():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('blog_Nourishing_Body.html',username=username)
    return render_template('index.html')

@app.route('/Importance_of_Games')
def Importance_of_Games():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('blog_Importance_of_Games.html',username=username)
    return render_template('index.html')

# Load your trained model
heart_disease_model = pickle.load(open(r"C:\Users\MSHigh School\Downloads\Remisense.ai-Webapp-main\Remisense.ai-Webapp-main\ML\heart_disease_model.sav", 'rb'))


@app.route('/heart-disease', methods=['GET', 'POST'])
def heart_disease():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = None
    user = User.query.get(session['user_id'])
    username = user.username

    
    if request.method == 'POST':
        try:
            # Collecting the form data
            age = int(request.form['age'])
            sex = request.form['sex']
            cp = request.form['cp']
            trtbps = int(request.form['trtbps'])
            chol = int(request.form['chol'])
            restecg = request.form['restecg']
            fbs = request.form['fbs']
            thalachh = int(request.form['thalachh'])
            oldpeak = float(request.form['oldpeak'])
            slp = request.form['slp']
            ca = int(request.form['ca'])
            thal = request.form['thal']
            exang = request.form['exang']

            # Preprocess the inputs
            # Encoding categorical variables into numerical values
            sex = 1 if sex.lower() == 'male' else 0
            cp = {'typical_angina': 0, 'atypical_angina': 1, 'non-anginal_pain': 2, 'asymptomatic': 3}.get(cp, 0)
            restecg = {'normal': 0, 'abnormalities': 1, 'probable_or_definite': 2}.get(restecg, 0)
            exang = 1 if exang.lower() == 'yes' else 0
            fbs = 1 if fbs.lower() == 'yes' else 0
            thal = {'normal': 0, 'fixed_defect': 1, 'reversable_defect': 2}.get(thal, 0)
            slp = {'upsloping': 0,'flat': 1,'downsloping': 2}.get(slp.lower(), 0)  

            # Create the feature array for prediction (ensure order and preprocessing match the model's training)
            features = np.array([age, sex, cp, trtbps, chol, restecg, thalachh, oldpeak, slp, ca, thal, exang, fbs]).reshape(1, -1)

            # Make prediction
            prediction = heart_disease_model.predict(features)
            result = "Positive for heart disease" if prediction[0] == 1 else "Negative for heart disease"

            return render_template('heart-disease.html', prediction=result, username=username)
            
        except Exception as e:
            # Handle errors
            return str(e)

    return render_template('heart-disease.html', prediction=None, username=username)


# Load your trained model
lung_cancer_model = pickle.load(open(r"C:\Users\MSHigh School\Downloads\Remisense.ai-Webapp-main\Remisense.ai-Webapp-main\ML\lung_cancer_model.sav", 'rb'))



@app.route('/lung-cancer', methods=['GET', 'POST'])
def lung_cancer():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = None
    user = User.query.get(session['user_id'])
    username = user.username

    if request.method == 'POST':
        try:
            # Collecting the form data
            GENDER = request.form['GENDER']
            AGE = int(request.form['AGE'])
            SMOKING = request.form['SMOKING']
            YELLOW_FINGERS = request.form['YELLOW_FINGERS']
            ANXIETY = request.form['ANXIETY']
            PEER_PRESSURE = request.form['PEER_PRESSURE']
            CHRONICDISEASE = request.form['CHRONICDISEASE']
            FATIGUE = request.form['FATIGUE']
            ALLERGY = request.form['ALLERGY']
            WHEEZING = request.form['WHEEZING']
            ALCOHOLCONSUMING = request.form['ALCOHOLCONSUMING']
            COUGHING = request.form['COUGHING']
            SHORTNESSOFBREATH = request.form['SHORTNESSOFBREATH']
            SWALLOWINGDIFFICULTY = request.form['SWALLOWINGDIFFICULTY']
            CHESTPAIN = request.form['CHESTPAIN']

            # Preprocess the inputs
            # Encoding categorical variables into numerical values
            GENDER = 1 if GENDER.lower() == 'male' else 0
            SMOKING = 1 if SMOKING.lower() == 'yes' else 0
            YELLOW_FINGERS = 1 if YELLOW_FINGERS.lower() == 'yes' else 0
            ANXIETY = 1 if ANXIETY.lower() == 'yes' else 0
            PEER_PRESSURE = 1 if PEER_PRESSURE.lower() == 'yes' else 0
            CHRONICDISEASE = 1 if CHRONICDISEASE.lower() == 'yes' else 0
            FATIGUE = 1 if FATIGUE.lower() == 'yes' else 0
            ALLERGY = 1 if ALLERGY.lower() == 'yes' else 0
            WHEEZING = 1 if WHEEZING.lower() == 'yes' else 0
            ALCOHOLCONSUMING = 1 if ALCOHOLCONSUMING.lower() == 'yes' else 0
            COUGHING = 1 if COUGHING.lower() == 'yes' else 0
            SHORTNESSOFBREATH = 1 if SHORTNESSOFBREATH.lower() == 'yes' else 0
            SWALLOWINGDIFFICULTY = 1 if SWALLOWINGDIFFICULTY.lower() == 'yes' else 0
            CHESTPAIN = 1 if CHESTPAIN.lower() == 'yes' else 0

            # Create the feature array for prediction
            features = np.array([GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONICDISEASE,
                                 FATIGUE, ALLERGY, WHEEZING, ALCOHOLCONSUMING, COUGHING, SHORTNESSOFBREATH,
                                 SWALLOWINGDIFFICULTY, CHESTPAIN]).reshape(1, -1)

            # Make prediction
            prediction = lung_cancer_model.predict(features)
            result = "Positive for lung cancer" if prediction[0] == 1 else "Negative for lung cancer"

            return render_template('lung-cancer.html', prediction=result, username=username)

        except Exception as e:
            # Handle errors
            return str(e)

    return render_template('lung-cancer.html', prediction=None, username=username)


    

@app.route('/admin')
def admin():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('admin.html',username=username)
    return render_template('index.html')

@app.route('/gemini_chat')
def gemini_chat():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('gemini.html',username=username)
    return render_template('index.html')

@app.route('/consultiva')
def consultiva():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('consultiva.html',username=username)
    return render_template('index.html')

@app.route('/videocall')
def videocall():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('videocall.html',username=username)
    
    return render_template('index.html')

@app.route('/ML')
def ML():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('ML.html',username=username)
    
    return render_template('index.html')

@app.route('/health-tracker')
def health_tracker():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('health-tracker.html',username=username)
    
    return render_template('index.html')

@app.route('/prediction-form')
def prediction_form():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('prediction-form.html',username=username)
    
    return render_template('index.html')

@app.route('/nearesthospital', methods=['GET', 'POST'])
def nearest_hospital():
    if request.method == 'GET':
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            username = user.username
            return render_template('nearesthospital.html', username=username)
        return render_template('index.html')
    elif request.method == 'POST':
        location = request.form['location']
        if not location:
            return jsonify({'error': 'Please provide a location'}), 400

        # Fetch coordinates for the entered location
        nominatim_url = f"https://nominatim.openstreetmap.org/search?format=json&q={location}&addressdetails=1&limit=1"
        response = requests.get(nominatim_url)
        if response.status_code != 200:
            return jsonify({'error': 'Error fetching coordinates'}), 500

        data = response.json()
        if not data:
            return jsonify({'error': 'Location not found'}), 404

        user_location = {
            'lat': float(data[0]['lat']),
            'lng': float(data[0]['lon'])
        }

        # Fetch nearby hospitals using user's location
        reverse_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={user_location['lat']}&lon={user_location['lng']}&zoom=18&addressdetails=1"
        response = requests.get(reverse_url)
        if response.status_code != 200:
            return jsonify({'error': 'Error fetching user location'}), 500

        data = response.json()
        city = data['address']['city']
        country = data['address']['country']

        hospitals_url = f"https://nominatim.openstreetmap.org/search?format=json&q=hospital&city={city}&country={country}&limit=10"
        response = requests.get(hospitals_url)
        if response.status_code != 200:
            return jsonify({'error': 'Error fetching hospitals'}), 500

        hospitals = response.json()

        return jsonify(hospitals)

@app.route('/nearestbloodbank', methods=['GET', 'POST'])
def nearest_bloodbank():
    if request.method == 'GET':
        # Assuming you're using session and User model
        # Make sure to import necessary modules and define User model
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            username = user.username
            return render_template('nearestbloodbank.html', username=username)
        return render_template('index.html')
    elif request.method == 'POST':
        location = request.form['location']
        if not location:
            return jsonify({'error': 'Please provide a location'}), 400

        # Fetch coordinates for the entered location
        nominatim_url = f"https://nominatim.openstreetmap.org/search?format=json&q={location}&addressdetails=1&limit=1"
        response = requests.get(nominatim_url)
        if response.status_code != 200:
            return jsonify({'error': 'Error fetching coordinates'}), 500

        data = response.json()
        if not data:
            return jsonify({'error': 'Location not found'}), 404

        user_location = {
            'lat': float(data[0]['lat']),
            'lng': float(data[0]['lon'])
        }

        # Fetch nearby blood banks using user's location
        reverse_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={user_location['lat']}&lon={user_location['lng']}&zoom=18&addressdetails=1"
        response = requests.get(reverse_url)
        if response.status_code != 200:
            return jsonify({'error': 'Error fetching user location'}), 500

        data = response.json()
        city = data['address']['city']
        country = data['address']['country']

        bloodbanks_url = f"https://nominatim.openstreetmap.org/search?format=json&q=bloodbank&city={city}&country={country}&limit=10"
        response = requests.get(bloodbanks_url)
        if response.status_code != 200:
            return jsonify({'error': 'Error fetching blood banks'}), 500

        bloodbanks = response.json()

        return jsonify(bloodbanks)


@app.route('/doctor-patients')
def doctor_patients():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username

        doctor = User.query.get(session['user_id'])

        if not doctor.type_of_doctor:
            return redirect(url_for('index'))

        # Fetch appointments assigned to the doctor
        appointments = Appointment.query.filter_by(type_of_doctor=doctor.type_of_doctor).all()
        file_list = os.listdir(text_files_dir)

        return render_template('doctor-patients.html', doctor=doctor, appointments=appointments,username=username,file_list=file_list)
    return render_template('index.html')

@app.route('/prescribe-medicine/<int:appointment_id>', methods=['GET', 'POST'])
def prescribe_medicine(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    doctor = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if appointment.type_of_doctor != doctor.type_of_doctor:
        return redirect(url_for('index'))

    available_medicines = ["Medicine 1", "Medicine 2", "Medicine 3"]  # Update this with your list of medicines

    if request.method == 'POST':
        selected_medicines = request.form.getlist('medicines[]')

        # Create a PDF document using ReportLab
        buffer = BytesIO()
        pdf = SimpleDocTemplate(buffer, pagesize=letter)

        # Define styles for the header and footer
        styles = getSampleStyleSheet()
        header_style = ParagraphStyle(
            'Header1',
            parent=styles['Heading1'],
            fontName='monospace',
            fontSize=18,
            spaceAfter=12,
            textColor=colors.green,
        )

        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
        )

        # Create content for the PDF
        content = []

        # Add Remisense header with green color
        Remisense_header = Paragraph("<font color='green' size='24'><b>Remisense: We Care for Your Health</b></font>", header_style)
        content.append(Remisense_header)

        # Add space after Remisense header
        content.append(Spacer(1, 12))

        # Add patient details
        patient_details = (
            f"<b>Patient Details:</b><br/>"
            f"Name: {appointment.name}<br/>"
            f"Age: {appointment.age}<br/>"
            f"Blood Group: {appointment.blood_group}<br/>"
            f"Phone Number: {appointment.phone_number}"
        )
        content.append(Paragraph(patient_details, styles['Normal']))

        # Add space after patient details
        content.append(Spacer(1, 12))

        # Add prescribed medicines
        prescribed_meds = "<b>Prescribed Medicines:</b><br/>"
        for medicine in selected_medicines:
            prescribed_meds += f"- {medicine}<br/>"
        content.append(Paragraph(prescribed_meds, styles['Normal']))

        # Add space after prescribed medicines
        content.append(Spacer(1, 12))

        # Add doctor details and footer
        doctor_details = (
            f"<b>Prescribed by Dr. {doctor.username} ({doctor.type_of_doctor})</b><br/>"
            "Thank you for choosing Remisense! We wish you good health."
        )
        content.append(Paragraph(doctor_details, styles['Normal']))

        # Add space after doctor details
        content.append(Spacer(1, 12))

        # Build the PDF
        pdf.build(content)

        # Save the PDF to the file
        pdf_filename = f"prescription_{appointment_id}.pdf"
        pdf_filepath = os.path.join("static", "prescriptions", pdf_filename)
        buffer.seek(0)
        with open(pdf_filepath, 'wb') as pdf_file:
            pdf_file.write(buffer.read())

        buffer.close()

        # Update appointment status to 'Prescribed'
        appointment.status = 'Prescribed'
        appointment.prescription_file = pdf_filepath
        db.session.commit()

        return redirect(url_for('doctor_patients'))

    return render_template('prescribe-medicine.html', appointment=appointment, available_medicines=available_medicines)
    
@app.route('/view-prescription/<int:appointment_id>')
def view_prescription(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    doctor = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if appointment.type_of_doctor != doctor.type_of_doctor or appointment.status != 'Prescribed':
        return redirect(url_for('index'))

    prescription_filepath = appointment.prescription_file

    return send_file(prescription_filepath, as_attachment=True)

@app.route('/view-prescription-patient/<int:appointment_id>')
def view_prescription_patient(appointment_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    appointment = Appointment.query.get(appointment_id)

    if not user or not appointment or appointment.user_id != user.id or appointment.status != 'Prescribed':
        return redirect(url_for('profile'))  # Change this line to redirect to the patient's profile instead of index

    # Read prescription text from the file
    prescription_filepath = appointment.prescription_file
    

    return send_file(prescription_filepath, as_attachment=True)

# ============================================================ scans ============================================================ 
    
@app.route('/braintumor', methods=['GET', 'POST'])
def braintumor():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('brain-tumor.html',username=username)
    else:
        return render_template('index.html')
    
@app.route('/alzheimer', methods=['GET', 'POST'])
def alzheimer():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('alzheimer.html',username=username)
    else:
        return render_template('index.html')
    
@app.route('/covid19', methods=['GET', 'POST'])
def covid19():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('covid19.html',username=username)
    else:
        return render_template('index.html')
    
@app.route('/malaria', methods=['GET', 'POST'])
def malaria():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('malaria.html',username=username)
    else:
        return render_template('index.html')
    
@app.route('/tuberculosis', methods=['GET', 'POST'])
def tuberculosis():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('tuberculosis.html',username=username)
    else:
        return render_template('index.html')
    

@app.route('/disease_predict', methods=['GET', 'POST'])
def disease_predict():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        chart_data={}
        if request.method == 'POST':
            selected_symptoms = []
            if(request.form['Symptom1']!="") and (request.form['Symptom1'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom1'])
            if(request.form['Symptom2']!="") and (request.form['Symptom2'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom2'])
            if(request.form['Symptom3']!="") and (request.form['Symptom3'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom3'])
            if(request.form['Symptom4']!="") and (request.form['Symptom4'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom4'])
            if(request.form['Symptom5']!="") and (request.form['Symptom5'] not in selected_symptoms):
                selected_symptoms.append(request.form['Symptom5'])
            disease, confidence_score = predict(selected_symptoms)
            
            chart_data = {
            'disease': disease,
            'confidence_score': confidence_score
            }
            return render_template('disease_predict.html',symptoms=symptoms,disease=disease, chart_data=chart_data,confidence_score=confidence_score,username=username)
            
        return render_template('disease_predict.html',symptoms=symptoms,username=username,chart_data=chart_data)
    else:
        return render_template('index.html')

@app.route('/pneumonia')
def pneumonia():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('pneumonia.html',username=username)
    else:
        return render_template('index.html')

@app.route('/eye')
def eye():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('eye.html',username=username)
    return render_template('index.html')

@app.route('/skin-cancer')
def skin_cancer():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username
        return render_template('skin-cancer.html',username=username)
    return render_template('index.html')


if __name__ == '__main__':
    create_tables()
    app.run(debug=True)