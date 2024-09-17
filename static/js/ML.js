const chatbox = document.getElementById('chatbox');
const userInput = document.getElementById('userInput');

const diseaseData = [
    // Flu and similar illnesses
    { symptoms: ['fever', 'headache', 'chills', 'muscle pain', 'fatigue', 'cough'], disease: 'Flu' },
    { symptoms: ['cough', 'shortness of breath', 'fever', 'sore throat', 'body aches', 'fatigue'], disease: 'COVID-19' },
    { symptoms: ['itching', 'skin rash', 'red patches', 'blisters'], disease: 'Fungal Infection' },
    
    // Allergies
    { symptoms: ['continuous sneezing', 'shivering', 'runny nose', 'itchy eyes', 'congestion'], disease: 'Allergy' },
    
    // GERD and related
    { symptoms: ['stomach pain', 'acidity', 'bloating', 'burping', 'nausea'], disease: 'GERD' },
    
    // Chronic cholestasis and drug reactions
    { symptoms: ['itching', 'vomiting', 'yellow skin', 'dark urine', 'abdominal pain'], disease: 'Chronic cholestasis' },
    { symptoms: ['itching', 'skin rash', 'stomach ache', 'fever', 'swelling'], disease: 'Drug Reaction' },
    
    // Peptic Ulcer Disease
    { symptoms: ['vomiting', 'indigestion', 'stomach pain', 'bloating', 'nausea'], disease: 'Peptic Ulcer Disease' },
    
    // AIDS
    { symptoms: ['high fever', 'muscle pain', 'weight loss', 'night sweats', 'swollen lymph nodes', 'chronic diarrhea'], disease: 'AIDS' },
    
    // Diabetes
    { symptoms: ['weight loss', 'restlessness', 'lethargy', 'frequent urination', 'excessive thirst', 'blurred vision'], disease: 'Diabetes' },
    
    // Gastroenteritis
    { symptoms: ['vomiting', 'sunken eyes', 'dehydration', 'diarrhoea', 'abdominal pain', 'fever'], disease: 'Gastroenteritis' },
    
    // Bronchial Asthma
    { symptoms: ['cough', 'high fever', 'breathlessness', 'wheezing', 'chest tightness', 'shortness of breath'], disease: 'Bronchial Asthma' },
    
    // Hypertension and Common Cold
    { symptoms: ['chest pain', 'headache', 'dizziness', 'shortness of breath', 'fatigue'], disease: 'Hypertension' },
    { symptoms: ['cough', 'high fever', 'headache', 'sneezing', 'sore throat', 'congestion'], disease: 'Common Cold' },
    
    // Jaundice
    { symptoms: ['high fever', 'weight loss', 'yellow skin', 'dark urine', 'itchy skin', 'abdominal pain'], disease: 'Jaundice' },
    
    // Migraine
    { symptoms: ['indigestion', 'headache', 'nausea', 'sensitivity to light', 'vomiting', 'visual disturbances'], disease: 'Migraine' },
    
    // Spondolities and Arthritis
    { symptoms: ['back pain', 'leg pain', 'stiffness', 'difficulty bending', 'reduced flexibility'], disease: 'Spondolities' },
    { symptoms: ['leg pain', 'joint stiffness', 'swelling', 'redness', 'warmth'], disease: 'Arthritis' },
    
    // Additional conditions
    { symptoms: ['shortness of breath', 'chest tightness', 'wheezing', 'chronic cough', 'difficulty breathing'], disease: 'Chronic Obstructive Pulmonary Disease (COPD)' },
    { symptoms: ['abdominal bloating', 'diarrhoea', 'weight loss', 'cramping', 'nausea'], disease: 'Irritable Bowel Syndrome (IBS)' },
    { symptoms: ['persistent cough', 'blood in sputum', 'weight loss', 'night sweats', 'fever'], disease: 'Tuberculosis (TB)' },
    { symptoms: ['joint pain', 'fever', 'rash', 'fatigue', 'swelling'], disease: 'Rheumatoid Arthritis' },
    { symptoms: ['fatigue', 'fever', 'swollen lymph nodes', 'night sweats', 'unexplained weight loss'], disease: 'Lymphoma' },
    { symptoms: ['sudden weight gain', 'swelling', 'high blood pressure', 'fatigue', 'shortness of breath'], disease: 'Kidney Disease' },
    { symptoms: ['severe headache', 'vision problems', 'nausea', 'vomiting', 'confusion'], disease: 'Stroke' },
    { symptoms: ['chronic cough', 'bloody sputum', 'shortness of breath', 'weight loss'], disease: 'Lung Cancer' },
    { symptoms: ['persistent diarrhea', 'abdominal cramps', 'weight loss', 'fever', 'nausea'], disease: 'Celiac Disease' },
    { symptoms: ['extreme fatigue', 'muscle pain', 'fever', 'joint pain', 'rash'], disease: 'Fibromyalgia' },
    { symptoms: ['dizziness', 'tinnitus', 'hearing loss', 'nausea', 'vomiting'], disease: 'Meniere\'s Disease' },
    { symptoms: ['persistent fever', 'chronic cough', 'unexplained weight loss', 'night sweats'], disease: 'Pneumonia' },
    { symptoms: ['extreme thirst', 'frequent urination', 'fatigue', 'blurred vision', 'slow wound healing'], disease: 'Type 2 Diabetes' },
    { symptoms: ['loss of appetite', 'weight loss', 'fatigue', 'abdominal pain', 'jaundice'], disease: 'Liver Cancer' },
    { symptoms: ['tingling in extremities', 'numbness', 'muscle weakness', 'difficulty walking'], disease: 'Multiple Sclerosis' },
    { symptoms: ['excessive thirst', 'frequent urination', 'weight loss', 'fatigue'], disease: 'Diabetes Mellitus' },
    { symptoms: ['abdominal pain', 'bloody stools', 'weight loss', 'persistent diarrhea'], disease: 'Crohn\'s Disease' }
    // Add more symptom-disease mappings as needed
];



function sendMessage() {
    const userMessage = userInput.value.trim();
    if (userMessage === '') return;

    // Display user message in the chatbox
    appendMessage('User', userMessage);

    // Process user symptoms and detect disease
    const detectedDisease = detectDisease(userMessage);

    // Display detected disease in the chatbox
    appendMessage('Chatbot', `Based on your symptoms, it could be ${detectedDisease}.`);

    // Clear user input
    userInput.value = '';
}

function detectDisease(userSymptoms) {
    // Convert user input to lowercase and split into an array of symptoms
    const symptomsArray = userSymptoms.toLowerCase().split(',');

    // Iterate through disease data to find a matching disease
    for (const data of diseaseData) {
        const intersection = data.symptoms.filter(symptom => symptomsArray.includes(symptom));
        if (intersection.length === data.symptoms.length) {
            return data.disease;
        }
    }

    // Return a default message if no matching disease is found
    return 'Any disease but we currently do not have enough information. Consult a healthcare professional for accurate diagnosis.';
}

function appendMessage(sender, message) {
    const messageElement = document.createElement('p');
    messageElement.textContent = `${sender}: ${message}`;
    chatbox.appendChild(messageElement);

    // Scroll to the bottom of the chatbox
    chatbox.scrollTop = chatbox.scrollHeight;
}
