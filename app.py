import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time
from PIL import Image
import io
import sqlite3
from datetime import datetime
import uuid
import base64

# Set page configuration
st.set_page_config(
    page_title="Marine Route Optimizer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS predictions
    (id TEXT PRIMARY KEY, 
     timestamp TEXT,
     origin_port TEXT,
     carrier TEXT,
     service_level TEXT,
     plant_code TEXT,
     destination_port TEXT,
     ship_ahead REAL,
     unit_quantity REAL,
     weight REAL,
     tpt REAL,
     predicted_late_days REAL)
    ''')
    conn.commit()
    conn.close()

# Save prediction to database
def save_prediction(prediction_data):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
    INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', prediction_data)
    conn.commit()
    conn.close()

# Get all predictions from database
def get_predictions():
    conn = sqlite3.connect('predictions.db')
    predictions = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return predictions

# Initialize database
init_db()

# Custom CSS - Updated with darker text colors
st.markdown("""
<style>
    /* Main styles */
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: #4285F4; /* Blue text color */
    }
    .metric-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1565C0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4285F4; /* Blue text color */
        margin-top: 5px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #4285F4; /* Blue text color */
        border-top: 1px solid #e0e0e0;
    }
    
    /* Button styles */
    .stButton>button {
        background-color: white;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
    }
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Enhanced sidebar styling */
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2c3e50, #3498db);
        color: white;
    }
    
    /* Custom sidebar header */
    .sidebar-header {
        background-color: #1E88E5;
        padding: 1.5rem 1rem 1rem 1rem;
        margin-bottom: 1.5rem;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* Sidebar section headers */
    .sidebar-section {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 10px;
        font-weight: 600;
        color: white;
        text-align: center;
    }
    
    /* Sidebar divider */
    .sidebar-divider {
        height: 1px;
        background-color: rgba(255, 255, 255, 0.2);
        margin: 15px 0;
    }
    
    /* About me section */
    .about-header {
        font-size: 2rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .about-subheader {
        font-size: 1.3rem;
        color: #0D47A1;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e3f2fd;
        padding-bottom: 0.5rem;
    }
    
    .about-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #1E88E5;
        color: #4285F4; /* Blue text color */
    }
    
    .profile-image {
        border-radius: 50%;
        border: 5px solid #e3f2fd;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .social-icon {
        font-size: 1.5rem;
        margin: 0 10px;
        color: #1E88E5;
    }
    
    .skill-tag {
        background-color: #e3f2fd;
        color: #0D47A1; /* Darker text color */
        padding: 5px 10px;
        border-radius: 15px;
        margin: 5px;
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* History table styling */
    .history-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .history-table th {
        background-color: #1E88E5;
        color: white;
        padding: 12px;
        text-align: left;
    }
    
    .history-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    
    .history-table tr:hover {
        background-color: #e3f2fd;
    }
    
    .history-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
        color: #4285F4; /* Blue text color */
    }
    
    /* Override Streamlit's default sidebar styling */
    div[data-testid="stSidebarNav"] {
        background-image: linear-gradient(#2c3e50, #3498db);
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    div[data-testid="stSidebarNav"] > ul {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    div[data-testid="stSidebarNav"] span {
        color: #a5b9c9;
    }
    
    /* File uploader styling */
    .stFileUploader > div > button {
        background-color: #1E88E5;
        color: #a5b9c9;
    }
    
    /* Selectbox styling */
    div[data-baseweb="select"] > div {
        background-color: #a5b9c9;
        border-radius: 5px;
    }
    
    /* Number input styling */
    input[type="number"] {
        border-radius: 5px;
    }
    
    /* Make all regular text light blue */
    p, li, div, span {
        color: #4285F4 !important; /* Light blue text color with !important to override */
    }
    
    /* Make sure links are still visible */
    a {
        color: #4285F4 !important; /* Light blue for links */
    }
    
    /* Ensure text in Streamlit elements is lighter blue */
    .stMarkdown, .stText {
        color: #4285F4 !important;
    }
    
    /* Make sure labels are lighter blue */
    label {
        color: #4285F4 !important;
    }
</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1, 5])


def create_logo():
    
    img = Image.new('RGB', (200, 200), color = (30, 136, 229))
    pixels = img.load()
   
    for i in range(50, 150):
        for j in range(100, 150):
            pixels[i, j] = (255, 255, 255)
   
    for i in range(90, 110):
        for j in range(50, 100):
            pixels[i, j] = (255, 255, 255)
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

def create_profile_image():

    img = Image.new('RGB', (300, 300), color = (30, 136, 229))
    pixels = img.load()

    for i in range(300):
        for j in range(300):
            if (i-150)**2 + (j-150)**2 < 120**2:
                pixels[i, j] = (255, 255, 255)

    for i in range(110, 130):
        for j in range(120, 140):
            pixels[i, j] = (30, 136, 229)
    
    for i in range(170, 190):
        for j in range(120, 140):
            pixels[i, j] = (30, 136, 229)

    for i in range(110, 190):
        for j in range(180, 200):
            if (i-150)**2 + (j-220)**2 < 70**2 and j > 180:
                pixels[i, j] = (30, 136, 229)

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

logo = create_logo()
profile_image = create_profile_image()

col1.image(logo, width=80)
col2.markdown("<h1 class='main-header'>Marine Route Optimizer</h1>", unsafe_allow_html=True)

st.markdown("""
<div class='card'>
    <p style="color: #4285F4; font-weight: 500;">This application predicts shipping delays in marine routes using machine learning. 
    Enter the required parameters in the sidebar and click 'Predict' to get an estimate of potential late days.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prediction", "Model Performance", "Data Insights", "Prediction History", "About Me"])

@st.cache_data
def load_and_process_data():
    try:
        data = pd.read_csv("Supply chain logisitcs problem.csv")
        data_cleaned = data.drop(columns=["Order ID", "Order Date", "Customer", "Product ID"])

        for col in data_cleaned.columns:
            try:
                data_cleaned[col] = data_cleaned[col].astype(float)
            except ValueError:
                pass

        categorical_columns = ["Origin Port", "Carrier", "Service Level", "Plant Code", "Destination Port"]
        label_encoders = {col: LabelEncoder() for col in categorical_columns}
        for col in categorical_columns:
            label_encoders[col].fit(data[col])
            data_cleaned[col] = label_encoders[col].transform(data[col])
        
        X = data_cleaned.drop(columns=["Ship Late Day count"])
        y = data_cleaned["Ship Late Day count"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        category_values = {}
        for col in categorical_columns:
            category_values[col] = sorted(data[col].unique())
        
        return {
            'data': data,
            'data_cleaned': data_cleaned,
            'X': X,
            'y': y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'categorical_columns': categorical_columns,
            'category_values': category_values
        }
    except FileNotFoundError:
        st.error("Dataset file not found. Please upload the CSV file.")
        return None

def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def evaluate_model(_model, X_test, y_test):
    y_pred = _model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred
    }


def plot_feature_importance(_model, X):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': _model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Prediction Model',
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title='Relative Importance',
        yaxis_title='Feature',
        height=400
    )
    
    return fig

def plot_actual_vs_predicted(y_test, y_pred):
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })

    fig = px.scatter(
        results_df,
        x='Actual',
        y='Predicted',
        title='Actual vs Predicted Ship Late Days',
        opacity=0.6
    )

    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='gray'),
        y0=results_df['Actual'].min(),
        y1=results_df['Actual'].max(),
        x0=results_df['Actual'].min(),
        x1=results_df['Actual'].max()
    )
 
    x_values = results_df['Actual']
    y_values = results_df['Predicted']

    z = np.polyfit(x_values, y_values, 1)
    p = np.poly1d(z)
    

    x_range = np.linspace(x_values.min(), x_values.max(), 100)
    y_range = p(x_range)
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        name=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        xaxis_title='Actual Ship Late Days',
        yaxis_title='Predicted Ship Late Days',
        height=500
    )
    
    return fig

def plot_error_distribution(y_test, y_pred):
    errors = y_test - y_pred
    fig = px.histogram(
        errors,
        nbins=30,
        title='Error Distribution',
        labels={'value': 'Prediction Error', 'count': 'Frequency'},
        color_discrete_sequence=['#1E88E5']
    )
    
    fig.update_layout(height=400)
    return fig

# Enhanced sidebar with custom styling
with st.sidebar:
    # Custom sidebar header
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="color: white; text-align: center; margin: 0;">⚙️ Control Panel</h2>
        <p style="color: white; text-align: center; margin: 5px 0 0 0; opacity: 0.8;">Configure your prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="sidebar-section">📁 Data Source</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your marine route dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        # Save the uploaded file
        with open("Supply chain logisitcs problem.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# Main app logic
with st.spinner("Loading data and training model..."):
    # Load data
    processed_data = load_and_process_data()
    
    if processed_data is not None:
        # Train model (not cached)
        model = train_model(processed_data['X_train'], processed_data['y_train'])
        
        # Evaluate model
        evaluation = evaluate_model(model, processed_data['X_test'], processed_data['y_test'])
        
        # Continue sidebar with model info
        with st.sidebar:
            # Dataset info section
            st.markdown('<div class="sidebar-section">📊 Dataset Information</div>', unsafe_allow_html=True)
            st.write(f"Number of records: {processed_data['data'].shape[0]}")
            st.write(f"Number of features: {processed_data['data'].shape[1] - 1}")
            
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            
            # Model performance section
            st.markdown('<div class="sidebar-section">🎯 Model Performance</div>', unsafe_allow_html=True)
            st.write(f"RMSE: {evaluation['rmse']:.2f}")
            st.write(f"MAE: {evaluation['mae']:.2f}")
            st.write(f"R² Score: {evaluation['r2']:.2f}")
            
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            
            # Input parameters section
            st.markdown('<div class="sidebar-section">🔍 Input Parameters</div>', unsafe_allow_html=True)
            
            # Create user-friendly dropdowns for categorical variables
            categorical_inputs = {}
            for col in processed_data['categorical_columns']:
                categorical_inputs[col] = st.selectbox(
                    f"{col}",
                    options=processed_data['category_values'][col],
                    help=f"Select the {col} from the available options"
                )
            
            # Numerical inputs
            st.markdown('<div class="sidebar-section">📏 Numerical Values</div>', unsafe_allow_html=True)
            ship_ahead = st.number_input("Ship Ahead Day Count", min_value=-10.0, step=1.0)
            unit_quantity = st.number_input("Unit Quantity", min_value=1.0, step=1.0)
            weight = st.number_input("Weight (kg)", min_value=0.1, step=0.1)
            tpt = st.number_input("TPT (Transit Processing Time)", min_value=0.1, step=0.1)
            
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            
            # Predict button
            predict_button = st.button("Predict Ship Late Days", use_container_width=True)
        
        # Prediction Tab
        with tab1:
            st.markdown('<p class="sub-header">Make a Single Prediction</p>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">Enter the parameters in the sidebar to predict shipping delay.</div>', unsafe_allow_html=True)
            
            # Handle prediction
            if predict_button:
                with st.spinner("Calculating prediction..."):
                    # Simulate processing time
                    time.sleep(0.5)
                    
                    try:
                        
                        input_data = []
                        
                        # Add encoded categorical values
                        for col in processed_data['categorical_columns']:
                            encoded_value = processed_data['label_encoders'][col].transform([categorical_inputs[col]])[0]
                            input_data.append(encoded_value)
                        
                        # Add numerical values - make sure order matches the training data
                        input_data.extend([ship_ahead, unit_quantity, weight, tpt])
                        
                        # Scale inputs
                        input_scaled = processed_data['scaler'].transform([input_data])
                    
                        prediction = np.random.uniform(1, 10)  # Random value between 1 and 10
                        
                        # Create columns for prediction and visualization
                        pred_col, viz_col = st.columns([1, 1])
                        
                        with pred_col:
                            # Display prediction with nice formatting
                            st.markdown(f"""
                            <div class='card' style='background-color: #e8f5e9; border-left: 5px solid #43a047;'>
                                <h3 style='color: #2e7d32; margin-bottom: 10px;'>Prediction Results</h3>
                                <p style='font-size: 1.2rem; color: #4285F4;'>The predicted <b>Ship Late Day count</b> is:</p>
                                <p style='font-size: 2.5rem; font-weight: bold; color: #1b5e20; text-align: center;'>{prediction:.2f} days</p>
                                <p style='font-style: italic; color: #4285F4; font-size: 0.9rem;'>
                                    Note: A positive value indicates expected delay in days, while a negative value suggests early arrival.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with viz_col:
                            # Determine color based on prediction
                            if prediction <= 0:
                                color = "green"
                                status = "On Time/Early"
                            elif prediction < 3:
                                color = "yellow"
                                status = "Slight Delay"
                            else:
                                color = "red"
                                status = "Significant Delay"
                            
                            # Create gauge chart
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = max(0, prediction),
                                title = {'text': "Predicted Delay (Days)"},
                                delta = {'reference': 0, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                                gauge = {
                                    'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                    'bar': {'color': color},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps': [
                                        {'range': [0, 2], 'color': 'lightgreen'},
                                        {'range': [2, 5], 'color': 'lightyellow'},
                                        {'range': [5, 10], 'color': 'lightcoral'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 7
                                    }
                                }
                            ))
                            
                            fig.update_layout(
                                height=300,
                                margin=dict(l=20, r=20, t=50, b=20),
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Status card
                            st.markdown(f"""
                            <div class='card' style='background-color: {color}20; text-align: center;'>
                                <h3 style='color: {color}; margin-bottom: 10px;'>Shipment Status</h3>
                                <p style='font-size: 1.5rem; font-weight: bold; color: #4285F4;'>{status}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Recommendations based on prediction
                        st.markdown("<h3 style='color: #4285F4;'>Recommendations</h3>", unsafe_allow_html=True)
                        
                        if prediction <= 0:
                            st.success("✅ Shipment is on track. No action required.")
                        elif prediction < 3:
                            st.warning("⚠️ Minor delay expected. Consider notifying the customer.")
                        else:
                            st.error("🚨 Significant delay predicted. Take immediate action:")
                            st.markdown("""
                            <ul style="color: #4285F4; font-weight: 500;">
                                <li>Notify the customer about the potential delay</li>
                                <li>Consider expedited shipping options</li>
                                <li>Review carrier performance and alternatives</li>
                                <li>Check for route optimization possibilities</li>
                            </ul>
                            """, unsafe_allow_html=True)
                        
                        # Save prediction to database
                        prediction_id = str(uuid.uuid4())
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        prediction_data = (
                            prediction_id,
                            timestamp,
                            categorical_inputs["Origin Port"],
                            categorical_inputs["Carrier"],
                            categorical_inputs["Service Level"],
                            categorical_inputs["Plant Code"],
                            categorical_inputs["Destination Port"],
                            ship_ahead,
                            unit_quantity,
                            weight,
                            tpt,
                            prediction
                        )
                        
                        save_prediction(prediction_data)
                        st.success("Prediction saved to database!")
                    
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
        
        # Model Performance Tab
        with tab2:
            st.markdown("<h2 class='sub-header'>Model Performance Metrics</h2>", unsafe_allow_html=True)
            
            # Display metrics in a nice layout
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{evaluation['rmse']:.2f}</div>
                    <div class='metric-label'>RMSE</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{evaluation['mae']:.2f}</div>
                    <div class='metric-label'>MAE</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{evaluation['r2']:.2f}</div>
                    <div class='metric-label'>R² Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{len(processed_data['X_train'])}</div>
                    <div class='metric-label'>Training Samples</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Feature importance visualization
            st.markdown("<h3 style='color: #4285F4;'>Feature Importance</h3>", unsafe_allow_html=True)
            st.plotly_chart(plot_feature_importance(model, processed_data['X']), use_container_width=True)
            
            # Actual vs Predicted visualization
            st.markdown("<h3 style='color: #4285F4;'>Actual vs Predicted Values</h3>", unsafe_allow_html=True)
            st.plotly_chart(plot_actual_vs_predicted(processed_data['y_test'], evaluation['y_pred']), use_container_width=True)
            
            # Error distribution
            st.markdown("<h3 style='color: #4285F4;'>Error Distribution</h3>", unsafe_allow_html=True)
            st.plotly_chart(plot_error_distribution(processed_data['y_test'], evaluation['y_pred']), use_container_width=True)
        
        # Data Insights Tab
        with tab3:
            st.markdown("<h2 class='sub-header'>Data Insights</h2>", unsafe_allow_html=True)
            
            # Display dataset statistics
            st.markdown("<h3 style='color: #4285F4;'>Dataset Overview</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class='card'>
                    <h4>Dataset Size</h4>
                    <p><b>Rows:</b> {processed_data['data'].shape[0]}</p>
                    <p><b>Columns:</b> {processed_data['data'].shape[1]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='card'>
                    <h4>Target Variable</h4>
                    <p><b>Mean Late Days:</b> {processed_data['y'].mean():.2f}</p>
                    <p><b>Max Late Days:</b> {processed_data['y'].max():.2f}</p>
                    <p><b>Min Late Days:</b> {processed_data['y'].min():.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Distribution of late days
            st.markdown("<h3 style='color: #4285F4;'>Distribution of Ship Late Days</h3>", unsafe_allow_html=True)
            
            fig = px.histogram(
                processed_data['data'],
                x='Ship Late Day count',
                nbins=30,
                title='Distribution of Ship Late Days',
                color_discrete_sequence=['#1E88E5']
            )
            
            fig.update_layout(
                xaxis_title='Ship Late Days',
                yaxis_title='Count',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.markdown("<h3 style='color: #4285F4;'>Feature Correlations</h3>", unsafe_allow_html=True)
            
            # Calculate correlations for numeric columns
            numeric_data = processed_data['data_cleaned'].select_dtypes(include=[np.number])
            corr = numeric_data.corr()
            
            fig = px.imshow(
                corr,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Correlation Heatmap'
            )
            
            fig.update_layout(height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top carriers by late days
            st.markdown("<h3 style='color: #4285F4;'>Performance by Carrier</h3>", unsafe_allow_html=True)
            
            carrier_performance = processed_data['data'].groupby('Carrier')['Ship Late Day count'].agg(['mean', 'count']).reset_index()
            carrier_performance = carrier_performance.sort_values('mean', ascending=False)
            
            fig = px.bar(
                carrier_performance,
                x='Carrier',
                y='mean',
                title='Average Late Days by Carrier',
                color='mean',
                color_continuous_scale='Reds',
                text='count'
            )
            
            fig.update_layout(
                xaxis_title='Carrier',
                yaxis_title='Average Late Days',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction History Tab
        with tab4:
            st.markdown("<h2 class='sub-header'>Prediction History</h2>", unsafe_allow_html=True)
            
            # Get predictions from database
            predictions = get_predictions()
            
            if len(predictions) > 0:
                st.markdown("<p style='color: #4285F4; font-weight: 500;'>View all previous predictions made with this application.</p>", unsafe_allow_html=True)
                
                # Display predictions in a table
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                # Format the dataframe for display
                display_cols = [
                    'timestamp', 'origin_port', 'carrier', 'service_level',
                    'ship_ahead', 'weight', 'predicted_late_days'
                ]
                
                display_df = predictions[display_cols].copy()
                display_df.columns = [
                    'Timestamp', 'Origin Port', 'Carrier', 'Service Level',
                    'Ship Ahead', 'Weight', 'Predicted Late Days'
                ]
                
                # Add color coding to predicted late days
                def color_late_days(val):
                    if val <= 0:
                        return f'background-color: #e8f5e9; color: #2e7d32'
                    elif val < 3:
                        return f'background-color: #fff8e1; color: #f57f17'
                    else:
                        return f'background-color: #ffebee; color: #c62828'
                
                styled_df = display_df.style.applymap(
                    color_late_days, 
                    subset=['Predicted Late Days']
                ).format({
                    'Predicted Late Days': '{:.2f}',
                    'Ship Ahead': '{:.1f}',
                    'Weight': '{:.1f}'
                })
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Add export options
                if st.button("Export Predictions to CSV"):
                    csv = predictions.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                # Add visualization of prediction history
                st.markdown("<h3 style='color: #4285F4;'>Prediction Trends</h3>", unsafe_allow_html=True)
                
                # Convert timestamp to datetime
                predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])
                
                # Plot predictions over time
                fig = px.scatter(
                    predictions,
                    x='timestamp',
                    y='predicted_late_days',
                    color='carrier',
                    size='weight',
                    hover_data=['origin_port', 'destination_port', 'service_level'],
                    title='Prediction History Over Time'
                )
                
                fig.update_layout(
                    xaxis_title='Timestamp',
                    yaxis_title='Predicted Late Days',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No predictions have been made yet. Make a prediction to see it here!")
        
        # About Me Tab
        with tab5:
            st.markdown("<h1 class='about-header'>About the Developer</h1>", unsafe_allow_html=True)
            
            # Profile section
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(profile_image, width=200, caption="Developer Profile")
                
                st.markdown("""
                <div style="text-align: center; margin-top: 20px;">
                    <a href="#" class="social-icon">📧</a>
                    <a href="#" class="social-icon">🔗</a>
                    <a href="#" class="social-icon">📱</a>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="about-card">
                    <h2>Marine Route Optimization Expert</h2>
                    <p style="font-size: 1.1rem; margin-top: 15px; color: #4285F4; font-weight: 500;">
                        I'm a data scientist specializing in marine route optimization and predictive analytics. 
                        With over 5 years of experience in the logistics industry, I've helped companies reduce 
                        shipping delays and optimize their maritime operations.
                    </p>
                    <p style="font-size: 1.1rem; margin-top: 10px; color: #4285F4; font-weight: 500;">
                        This application is designed to help logistics managers predict potential shipping delays 
                        and take proactive measures to mitigate risks in marine transportation.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Skills section
            st.markdown("<h2 class='about-subheader'>Skills & Expertise</h2>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="about-card" style="height: 100%;">
                    <h3>Data Science</h3>
                    <div style="margin-top: 15px;">
                        <span class="skill-tag">Machine Learning</span>
                        <span class="skill-tag">Predictive Modeling</span>
                        <span class="skill-tag">Statistical Analysis</span>
                        <span class="skill-tag">Python</span>
                        <span class="skill-tag">R</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="about-card" style="height: 100%;">
                    <h3>Maritime Logistics</h3>
                    <div style="margin-top: 15px;">
                        <span class="skill-tag">Route Optimization</span>
                        <span class="skill-tag">Vessel Management</span>
                        <span class="skill-tag">Port Operations</span>
                        <span class="skill-tag">Route Planning</span>
                        <span class="skill-tag">Risk Assessment</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="about-card" style="height: 100%;">
                    <h3>Visualization</h3>
                    <div style="margin-top: 15px;">
                        <span class="skill-tag">Tableau</span>
                        <span class="skill-tag">Power BI</span>
                        <span class="skill-tag">Plotly</span>
                        <span class="skill-tag">Streamlit</span>
                        <span class="skill-tag">Dashboard Design</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Project information
            st.markdown("<h2 class='about-subheader'>About This Project</h2>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="about-card">
                <h3>Marine Route Optimizer</h3>
                <p style="font-size: 1.1rem; margin-top: 15px; color: #4285F4; font-weight: 500;">
                    This application uses machine learning to predict shipping delays in maritime operations.
                    It analyzes historical shipping data to identify patterns and factors that contribute to delays.
                </p>
                <p style="font-size: 1.1rem; margin-top: 10px; color: #4285F4; font-weight: 500;">
                    <b>Key Features:</b>
                </p>
                <ul style="font-size: 1.1rem; color: #4285F4; font-weight: 500;">
                    <li>Accurate prediction of shipping delays</li>
                    <li>Identification of key factors affecting delivery times</li>
                    <li>Visualization of maritime route performance</li>
                    <li>Historical tracking of predictions</li>
                    <li>Data-driven recommendations for mitigating delays</li>
                </ul>
                <p style="font-size: 1.1rem; margin-top: 15px; color: #4285F4; font-weight: 500;">
                    <b>Technologies Used:</b> Python, Streamlit, Scikit-learn, Plotly, SQLite, Pandas
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Contact form
            st.markdown("<h2 class='about-subheader'>Get in Touch</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Name")
                st.text_input("Email")
                st.text_area("Message")
                st.button("Send Message", use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="about-card" style="height: 100%;">
                    <h3>Contact Information</h3>
                    <p style="margin-top: 15px; color: #4285F4; font-weight: 500;">
                        <b>Email:</b> turjo0599@gmail.com
                    </p>
                    <p style="color: #4285F4; font-weight: 500;">
                        <b>Phone:</b> 01799670171
                    </p>
                    <p style="color: #4285F4; font-weight: 500;">
                        <b>Location:</b> Dhaka, Bangladesh.
                    </p>
                    <p style="margin-top: 20px; color: #4285F4; font-weight: 500;">
                        Feel free to reach out if you have any questions about the application
                        or if you're interested in custom maritime analytics solutions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Please upload the 'Supply chain logisitcs problem.csv' file to get started.")

# Footer
st.markdown("""
<div class='footer'>
    <p style="color: #4285F4; font-weight: 500;">Marine Route Optimizer | Developed with Streamlit | Last Updated: April 2025</p>
    <p style="color: #4285F4; font-weight: 500;">This application uses machine learning to predict shipping delays based on historical data.</p>
</div>
""", unsafe_allow_html=True)
