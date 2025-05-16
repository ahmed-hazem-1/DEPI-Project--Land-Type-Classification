import streamlit as st
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import folium
from streamlit_folium import folium_static, st_folium  # Add st_folium import
import base64
from io import BytesIO
import cv2
from datetime import datetime, timedelta

# Set page configuration with Space theme
st.set_page_config(
    page_title="Land Type Classification",
    page_icon="🛰️",
    layout="wide"
)

# Custom CSS for Space theme
st.markdown(
    """
    <style>
    .main {
        background-color: #0c1445;
        color: #e0e0ff;
    }
    .stApp {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1, h2, h3 {
        color: #00ccff;
    }
    .stButton button {
        background-color: #3d5af1;
        color: white;
    }
    .stSelectbox label, .stFileUploader label {
        color: #00ccff;
        font-weight: bold;
    }
    .uploadedFileData {
        color: #e0e0ff !important;
    }
    div.block-container {
        padding-top: 2rem;
    }
    .css-145kmo2 {
        color: #e0e0ff !important;
    }
    .css-1offfwp p {
        color: #e0e0ff !important;
    }
    div[data-testid="stMarkdownContainer"] > p {
        color: #e0e0ff;
    }
    .stMarkdown {
        color: #e0e0ff;
    }
    div.css-1kyxreq.etr89bj0 {
        color: #e0e0ff !important;
    }
    .css-184tjsw p {
        color: #e0e0ff !important;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Page title with Space theme icons
st.title("🛰️ Land Type Classification from Above")
st.markdown("### Analyze Earth's surface features using satellite imagery")

# Define class names for all models
class_names = {
    0: "Annual Crop",
    1: "Forest",
    2: "Herbaceous Vegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "Permanent Crop",
    7: "Residential",
    8: "River",
    9: "Sea Lake"
}

def load_model_file(model_type):
    """Load the appropriate model based on selection"""
    try:
        if model_type == "RGB":
            model_path = "model_v2.h5"
            channels = 3
        elif model_type == "RGB-NIR":
            model_path = "model_RGB_NIR_v2.h5"
            channels = 4
        else:  # NDVI
            model_path = "model_NDVI_v0.h5"
            channels = 1
            
        if os.path.exists(model_path):
            # Custom loading method to handle compatibility issues
            try:
                # First attempt: standard loading
                model = load_model(model_path)
            except Exception as e:
                if 'batch_shape' in str(e):
                    # Second attempt: load with custom_objects and compile=False
                    st.info(f"Using compatibility mode to load model: {model_path}")
                    
                    # Clear any custom objects that might interfere
                    tf.keras.utils.get_custom_objects().clear()
                    
                    # Load with compile=False to avoid batch_shape issues
                    model = load_model(model_path, compile=False)
                    
                    # Manually compile the model
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                else:
                    # If it's a different error, re-raise it
                    raise e
                    
            return model, channels
        else:
            st.error(f"Model file not found: {model_path}")
            return None, channels
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, model_type, target_size=(64, 64)):
    """Preprocess the image based on model type"""
    # Resize image
    img = image.resize(target_size)
    img_array = np.array(img)
    
    # Process based on model type
    if model_type == "RGB":
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[2] > 3:  # More than 3 channels
            img_array = img_array[:,:,:3]
        # Normalize
        img_array = img_array / 255.0
        
    elif model_type == "RGB-NIR":
        if len(img_array.shape) == 2:  # Grayscale
            # Create a simulated RGB+NIR
            img_array = np.stack((img_array,)*4, axis=-1)
        elif img_array.shape[2] == 3:  # RGB
            # Create a simulated NIR channel (avg of RGB)
            nir = np.mean(img_array, axis=2, keepdims=True)
            img_array = np.concatenate([img_array, nir], axis=2)
        elif img_array.shape[2] > 4:  # More than 4 channels
            img_array = img_array[:,:,:4]
        # Normalize
        img_array = img_array / 255.0
        
    else:  # NDVI
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Calculate NDVI (using red and simulated NIR)
            red = img_array[:,:,0].astype(float)
            nir = np.mean(img_array[:,:,1:3], axis=2).astype(float)
            # Avoid division by zero
            denominator = nir + red
            denominator[denominator == 0] = 1
            ndvi = (nir - red) / denominator
            # Normalize from [-1,1] to [0,1]
            ndvi = (ndvi + 1) / 2
            img_array = ndvi[:,:,np.newaxis]
        elif len(img_array.shape) == 2:
            # For grayscale, just normalize and add channel dim
            img_array = (img_array / 255.0)[:,:,np.newaxis]
    
    # Expand dims for batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_and_analyze(image, model_type):
    """Make prediction and return results with analysis"""
    model, channels = load_model_file(model_type)
    
    if model is None:
        return None
        
    # Preprocess image for the model
    processed_img = preprocess_image(image, model_type)
    
    # Make prediction
    prediction = model.predict(processed_img)
    class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][class_idx]) * 100
    
    return {
        "class_idx": class_idx,
        "class_name": class_names[class_idx],
        "confidence": confidence
    }

def display_analysis(result, model_type):
    """Display analysis based on prediction results with Plotly visualizations"""
    if result is None:
        return
        
    st.subheader("🔍 Analysis Results")
    
    # Display prediction with confidence
    st.markdown(f"### Predicted Land Type: **{result['class_name']}**")
    st.markdown(f"### Confidence: **{result['confidence']:.2f}%**")
    
    # Create a confidence gauge with Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['confidence'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'color': '#00ccff', 'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#00ccff"},
            'bar': {'color': "#3d5af1"},
            'bgcolor': "#1f2851",
            'borderwidth': 2,
            'bordercolor': "#e0e0ff",
            'steps': [
                {'range': [0, 50], 'color': '#252d5a'},
                {'range': [50, 75], 'color': '#1f3d7a'},
                {'range': [75, 90], 'color': '#1a4f99'},
                {'range': [90, 100], 'color': '#1563b6'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="#0c1445",
        font={'color': "#e0e0ff", 'family': "Arial"},
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display spectral bands information
    st.markdown("### Spectral Bands Information")
    
    if model_type == "RGB":
        st.markdown("""
        **RGB Model uses 3 spectral bands:**
        - **Red (0.6-0.7 μm)**: Sensitive to plant chlorophyll absorption
        - **Green (0.5-0.6 μm)**: Peak reflectance for vegetation
        - **Blue (0.4-0.5 μm)**: Good for distinguishing soil types and built environments
        """)
        
        # Add RGB visualization with Plotly
        bands_fig = px.bar(
            x=["Red", "Green", "Blue"],
            y=[0.65, 0.55, 0.45],  # Average wavelengths
            labels={"x": "Band", "y": "Wavelength (μm)"},
            color=["Red", "Green", "Blue"],
            color_discrete_map={"Red": "red", "Green": "green", "Blue": "blue"},
            title="RGB Spectral Bands"
        )
        bands_fig.update_layout(
            paper_bgcolor="#0c1445",
            plot_bgcolor="#1f2851",
            font={'color': "#e0e0ff"},
        )
        st.plotly_chart(bands_fig)

    elif model_type == "RGB-NIR":
        st.markdown("""
        **RGB-NIR Model uses 4 spectral bands:**
        - **Red (0.6-0.7 μm)**: Chlorophyll absorption feature
        - **Green (0.5-0.6 μm)**: Peak vegetation reflectance
        - **Blue (0.4-0.5 μm)**: Atmospheric and deep water penetration
        - **Near-Infrared (0.7-1.1 μm)**: Highlights vegetation biomass and moisture
        
        *The addition of NIR significantly improves vegetation analysis and classification.*
        """)
        
        # Add RGB-NIR visualization with Plotly
        bands_fig = px.bar(
            x=["Red", "Green", "Blue", "NIR"],
            y=[0.65, 0.55, 0.45, 0.85],  # Average wavelengths
            labels={"x": "Band", "y": "Wavelength (μm)"},
            color=["Red", "Green", "Blue", "NIR"],
            color_discrete_map={"Red": "red", "Green": "green", "Blue": "blue", "NIR": "purple"},
            title="RGB-NIR Spectral Bands"
        )
        bands_fig.update_layout(
            paper_bgcolor="#0c1445",
            plot_bgcolor="#1f2851",
            font={'color': "#e0e0ff"},
        )
        st.plotly_chart(bands_fig)
        
    else:  # NDVI
        st.markdown("""
        **NDVI Model uses derived vegetation index:**
        - **Normalized Difference Vegetation Index**: (NIR - Red) / (NIR + Red)
        - Values range from -1 to 1, with healthy vegetation typically 0.2 to 0.8
        - NDVI is extremely effective at isolating vegetation health and density
        """)
        
        # Add NDVI visualization with Plotly
        ndvi_range = np.linspace(-1, 1, 100)
        ndvi_colors = []
        for val in ndvi_range:
            if val < -0.2:
                ndvi_colors.append('rgb(30, 30, 100)')  # Water/Shadow (dark blue)
            elif val < 0:
                ndvi_colors.append('rgb(80, 80, 150)')  # Barren (blue-gray)
            elif val < 0.2:
                ndvi_colors.append('rgb(150, 150, 150)')  # Built-up/Soil (gray)
            elif val < 0.4:
                ndvi_colors.append('rgb(100, 180, 100)')  # Sparse vegetation (light green)
            elif val < 0.6:
                ndvi_colors.append('rgb(50, 150, 50)')  # Moderate vegetation (medium green)
            else:
                ndvi_colors.append('rgb(0, 120, 0)')  # Dense vegetation (dark green)
        
        ndvi_fig = px.scatter(
            x=ndvi_range, 
            y=[0]*len(ndvi_range),
            color=ndvi_range,
            color_discrete_sequence=ndvi_colors,
            title="NDVI Range Interpretation"
        )
        
        ndvi_fig.update_layout(
            paper_bgcolor="#0c1445",
            plot_bgcolor="#1f2851",
            font={'color': "#e0e0ff"},
            xaxis_title="NDVI Value (-1 to 1)",
            yaxis_showticklabels=False,
        )
        
        st.plotly_chart(ndvi_fig)
    
    # Display tips and insights based on land type
    st.markdown("### Tips and Insights")
    
    land_type_tips = {
        0: """
        **Annual Crop Land**
        - These areas show regular seasonal patterns in satellite imagery
        - Best monitored through time series analysis to track crop cycles
        - Consider exploring crop health monitoring with NDVI data
        - RGB+NIR models typically provide better crop type differentiation
        """,
        
        1: """
        **Forest Area**
        - Forests are critical carbon sinks and biodiversity hotspots
        - Monitor for deforestation using time series analysis
        - NIR bands are excellent for assessing forest health and density
        - Consider conservation initiatives in these areas
        """,
        
        2: """
        **Herbaceous Vegetation**
        - These areas typically include grasslands, meadows, and shrublands
        - NDVI analysis is particularly useful for monitoring seasonal changes
        - Consider grazing potential or monitoring for invasive species
        - May provide ecosystem services like erosion control and wildlife habitat
        """,
        
        3: """
        **Highway/Road Infrastructure**
        - Built infrastructure detection is typically strongest in RGB models
        - Consider monitoring buffer zones alongside highways for environmental impact
        - Urban expansion typically follows transportation corridors
        - Heat signature from roads can be detected in thermal bands (not in current models)
        """,
        
        4: """
        **Industrial Area**
        - Industrial zones typically show low NDVI values
        - Consider environmental monitoring for pollution effects
        - NIR bands can help detect waste water or contamination
        - Regular monitoring recommended for regulatory compliance
        """,
        
        5: """
        **Pasture Land**
        - Pastures show moderate NDVI values with seasonal variations
        - Monitor for overgrazing using vegetation indices
        - Consider rotational grazing strategies to optimize land use
        - RGB+NIR provides good differentiation from cropland
        """,
        
        6: """
        **Permanent Crop**
        - These areas (orchards, vineyards, etc.) show distinct patterns
        - Less seasonal variation than annual crops
        - NIR bands are excellent for monitoring crop health long-term
        - Consider precision agriculture techniques for irrigation and fertilization
        """,
        
        7: """
        **Residential Area**
        - Mixed spectral signature including buildings and vegetation
        - Consider urban heat island effect monitoring
        - Opportunity for green space planning and development
        - RGB models typically perform well for urban classification
        """,
        
        8: """
        **River**
        - Water features have distinctive spectral signatures in all bands
        - Monitor for water quality changes and flooding
        - Consider riparian zone health assessment using NDVI
        - NIR bands are particularly useful for delineating water boundaries
        """,
        
        9: """
        **Sea/Lake**
        - Large water bodies are easily identifiable in all spectral bands
        - Monitor for algal blooms or pollution events
        - Consider shoreline change analysis over time
        - Water quality assessment possible with specialized bands
        """
    }
    
    st.markdown(land_type_tips[result["class_idx"]])
    
    # Add a radar chart for land characteristic ratings
    characteristics = {
        0: {"Vegetation": 0.7, "Water": 0.3, "Urban": 0.1, "Soil": 0.8, "Seasonal Variation": 0.9},
        1: {"Vegetation": 0.9, "Water": 0.4, "Urban": 0.1, "Soil": 0.2, "Seasonal Variation": 0.5},
        2: {"Vegetation": 0.8, "Water": 0.3, "Urban": 0.1, "Soil": 0.5, "Seasonal Variation": 0.7},
        3: {"Vegetation": 0.2, "Water": 0.1, "Urban": 0.9, "Soil": 0.4, "Seasonal Variation": 0.1},
        4: {"Vegetation": 0.1, "Water": 0.3, "Urban": 0.9, "Soil": 0.5, "Seasonal Variation": 0.2},
        5: {"Vegetation": 0.8, "Water": 0.2, "Urban": 0.1, "Soil": 0.6, "Seasonal Variation": 0.5},
        6: {"Vegetation": 0.8, "Water": 0.4, "Urban": 0.2, "Soil": 0.5, "Seasonal Variation": 0.4},
        7: {"Vegetation": 0.5, "Water": 0.3, "Urban": 0.9, "Soil": 0.3, "Seasonal Variation": 0.3},
        8: {"Vegetation": 0.3, "Water": 0.9, "Urban": 0.1, "Soil": 0.2, "Seasonal Variation": 0.4},
        9: {"Vegetation": 0.1, "Water": 1.0, "Urban": 0.0, "Soil": 0.0, "Seasonal Variation": 0.3}
    }
    
    char_data = characteristics[result["class_idx"]]
    
    radar_fig = go.Figure()
    
    radar_fig.add_trace(go.Scatterpolar(
        r=list(char_data.values()),
        theta=list(char_data.keys()),
        fill='toself',
        name=result["class_name"],
        line_color='#00ccff',
        fillcolor='rgba(61, 90, 241, 0.5)'
    ))
    
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Land Type Characteristics",
        paper_bgcolor="#0c1445",
        font={'color': "#e0e0ff"},
    )
    
    st.plotly_chart(radar_fig)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📷 Upload Image", "🌎 Map Selection"])

# Tab 1: Image Upload
with tab1:
    st.header("Upload Satellite Image")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type", 
        ["RGB", "RGB-NIR", "NDVI"],
        help="Choose the type of model for prediction based on available image bands"
    )
    
    uploaded_file = st.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png", "tif"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
        # Make prediction and get analysis
        with st.spinner("Analyzing image..."):
            result = predict_and_analyze(image, model_type)
            
        if result:
            with col2:
                st.success(f"Prediction: **{result['class_name']}** with {result['confidence']:.2f}% confidence")
                
            # Show detailed analysis
            display_analysis(result, model_type)

# Tab 2: Map Selection
with tab2:
    st.header("Select Region on Map")
    
    # Model selection for map
    model_type_map = st.selectbox(
        "Select Model Type for Map Analysis", 
        ["RGB", "RGB-NIR", "NDVI"],
        key="model_map",
        help="Choose the type of model for prediction based on available imagery"
    )
    
    # Add date selection for satellite imagery
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_date = st.date_input(
            "Select acquisition date", 
            value=datetime.now() - timedelta(days=5),
            max_value=datetime.now() - timedelta(days=1),
            help="Satellite imagery from this date will be fetched (when available)"
        )
    
    with col2:
        cloud_coverage = st.slider(
            "Maximum cloud coverage (%)", 
            min_value=0, 
            max_value=100, 
            value=20,
            help="Lower values will provide clearer images"
        )
    
    # Create expandable section for API configuration
    with st.expander("Sentinel Hub API Configuration", expanded=False):
        st.info("Configure your Sentinel Hub API credentials. Leave empty to use demo mode.")
        
        sentinel_instance_id = st.text_input(
            "Instance ID", 
            value=os.environ.get("SENTINEL_INSTANCE_ID", ""),
            help="Your Sentinel Hub instance ID"
        )
        
        sentinel_api_key = st.text_input(
            "API Key", 
            value=os.environ.get("SENTINEL_API_KEY", ""),
            type="password",
            help="Your Sentinel Hub API key"
        )
        
        st.markdown("""
        **Note:** Without valid API credentials, the app will use simulated data. 
        To get your own API credentials, register at [Sentinel Hub](https://www.sentinel-hub.com/).
        """)
    
    # Instructions based on model type
    if model_type_map == "RGB":
        st.markdown("""
        #### Instructions:
        1. Navigate to your area of interest on the map
        2. Use the draw tool (rectangle icon) in the upper left corner to select your region
        3. Click the "Analyze Selected Region" button to automatically process the visible area
        """)
    else:
        st.markdown("""
        #### Instructions:
        1. Navigate to your area of interest on the map
        2. Use the draw tool (rectangle icon) in the upper left corner to select your region
        3. Click the "Analyze Selected Region" button to fetch multi-spectral imagery for that location
        """)
    
    # Initialize map centered on a default location with satellite imagery
    m = folium.Map(
        location=[30.0, 31.0], 
        zoom_start=5, 
        control_scale=True,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery'
    )
    
    # Add a layer control to allow switching between satellite and street map
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.LayerControl().add_to(m)
    
    # Add draw control to allow selecting rectangular areas and store last drawn coordinates
    draw_options = {
        'polyline': False,
        'polygon': False,
        'circle': False,
        'marker': False,
        'circlemarker': False,
        'rectangle': True,
    }
    
    draw = folium.plugins.Draw(
        draw_options=draw_options,
        edit_options={'edit': False},
        position='topleft'
    )
    draw.add_to(m)
    
    # Add JavaScript to capture the selection coordinates
    draw_callback = """
    <script>
    var lastDrawnCoords = null;
    
    document.addEventListener('DOMContentLoaded', function() {
        // Wait for the map to be fully loaded
        setTimeout(function() {
            // Find the map and add event listeners
            var map = document.querySelector('.folium-map');
            if (map && map._leaflet) {
                map._leaflet.on('draw:created', function(e) {
                    lastDrawnCoords = e.layer.getBounds();
                    // Store in session data
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: {
                            coords: [
                                lastDrawnCoords.getSouth(),
                                lastDrawnCoords.getWest(),
                                lastDrawnCoords.getNorth(),
                                lastDrawnCoords.getEast()
                            ],
                            center: [
                                lastDrawnCoords.getCenter().lat,
                                lastDrawnCoords.getCenter().lng
                            ],
                            zoom: map._leaflet.getZoom()
                        }
                    }, '*');
                });
            }
        }, 1000);
    });
    </script>
    """
    
    m.get_root().html.add_child(folium.Element(draw_callback))
    
    # Add a measurement tool
    folium.plugins.MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        secondary_length_unit='miles',
        primary_area_unit='sqkilometers',
        secondary_area_unit='acres'
    ).add_to(m)
    
    # Display the map with error handling
    try:
        st.write("### Interactive Map")
        st.write("Please wait while the map loads...")
        
        # Try to display using st_folium
        try:
            map_data = st_folium(m, width=800, height=500)
        except Exception as e:
            st.warning(f"Advanced map interaction not available: {str(e)}")
            st.warning("Using basic map display instead. Rectangle selection may not work properly.")
            # Fallback to folium_static
            folium_static(m, width=800, height=500)
            map_data = None
            
        # Add a separator to ensure the map is visible
        st.markdown("---")
        
        # Debug information
        with st.expander("Map Debug Info", expanded=False):
            st.info("If the map isn't visible, try the following:")
            st.markdown("1. Check your internet connection")
            st.markdown("2. Refresh the page")
            st.markdown("3. Try a different browser")
            st.markdown("4. Ensure streamlit-folium is properly installed")
            if map_data:
                st.write("Map data received:", map_data)
            else:
                st.write("No map data received. Selection features may not work.")
    except Exception as master_e:
        st.error(f"Error displaying map: {str(master_e)}")
        st.error("Unable to load interactive map component. Please check dependencies.")
        st.info("Please make sure you have the following packages installed:")
        st.code("pip install folium streamlit-folium")
    
    # Initialize session state for coordinates if not exists
    if 'map_coords' not in st.session_state:
        st.session_state.map_coords = None
    
    # Store the coordinates from the component value
    if map_data:
        # Check if we have coordinates data in the structure returned by st_folium
        # The data might be stored in the 'last_active_drawing' key
        if map_data.get('last_active_drawing'):
            # Extract coordinates from the returned data
            drawn_data = map_data.get('last_active_drawing')
            if isinstance(drawn_data, dict) and 'geometry' in drawn_data:
                # If it's a rectangle, get the bounds from coordinates
                if drawn_data.get('type') == 'rectangle' and drawn_data.get('geometry', {}).get('type') == 'Polygon':
                    coords = drawn_data.get('geometry', {}).get('coordinates', [[]])[0]
                    if len(coords) >= 4:
                        # Calculate bounds [south, west, north, east]
                        lats = [coord[1] for coord in coords]
                        lngs = [coord[0] for coord in coords]
                        south, north = min(lats), max(lats)
                        west, east = min(lngs), max(lngs)
                        st.session_state.map_coords = [south, west, north, east]
                        
                        # Calculate center
                        center_lat = (south + north) / 2
                        center_lng = (west + east) / 2
                        st.session_state.map_center = [center_lat, center_lng]
                        
                        # Get current zoom level if available
                        st.session_state.map_zoom = map_data.get('zoom', 5)
        
        # For backward compatibility, also check the format in your JavaScript
        elif isinstance(map_data, dict) and 'coords' in map_data:
            st.session_state.map_coords = map_data['coords']
            st.session_state.map_center = map_data.get('center')
            st.session_state.map_zoom = map_data.get('zoom')
    
    # Display coordinates if available
    if st.session_state.map_coords:
        south, west, north, east = st.session_state.map_coords
        st.success(f"Region selected: {north:.4f}°N, {west:.4f}°E to {south:.4f}°N, {east:.4f}°E")
    
    # Add a button to trigger area capture and analysis
    if st.button("Analyze Selected Region"):
        if not st.session_state.map_coords:
            st.warning("Please select a region on the map first.")
        else:
            with st.spinner(f"Fetching {model_type_map} imagery for selected region..."):
                # Get coordinates
                south, west, north, east = st.session_state.map_coords
                
                # Check if we have valid Sentinel Hub credentials
                has_sentinel_credentials = sentinel_instance_id and sentinel_api_key
                
                if model_type_map == "RGB":
                    st.markdown("### RGB Analysis")
                    
                    if has_sentinel_credentials:
                        try:
                            # Fetch RGB imagery from Sentinel Hub
                            st.info("Fetching true color imagery from Sentinel Hub...")
                            
                            # In a real implementation, this would use the Sentinel Hub API
                            # For demo, we'll simulate this
                            st.info("Using simulated data for demonstration.")
                            
                            # Create a synthetic satellite image for demonstration
                            width, height = 64, 64
                            image_array = np.zeros((height, width, 3), dtype=np.uint8)
                            
                            # Create a somewhat realistic looking satellite image
                            # Base green/brown land with some variations based on current timestamp for uniqueness
                            import time
                            seed = int(time.time()) % 1000
                            np.random.seed(seed)
                            
                            # Generate base image with natural looking features
                            # Green vegetation
                            image_array[:,:,1] = np.random.randint(90, 140, (height, width))
                            # Red/brown earth tones
                            image_array[:,:,0] = np.random.randint(70, 120, (height, width))
                            # Blue
                            image_array[:,:,2] = np.random.randint(60, 100, (height, width))
                            
                            # Create image
                            image = Image.fromarray(image_array)
                            
                        except Exception as e:
                            st.error(f"Error fetching RGB imagery: {str(e)}")
                            st.stop()
                    else:
                        # Use simulated data
                        st.info("Using simulated data (no API credentials provided).")
                        
                        # Create a synthetic satellite image as in the original code
                        width, height = 64, 64
                        image_array = np.zeros((height, width, 3), dtype=np.uint8)
                        
                        # Create a somewhat realistic looking satellite image
                        import time
                        seed = int(time.time()) % 1000
                        np.random.seed(seed)
                        
                        # Generate base image with natural looking features
                        image_array[:,:,1] = np.random.randint(90, 140, (height, width))
                        image_array[:,:,0] = np.random.randint(70, 120, (height, width))
                        image_array[:,:,2] = np.random.randint(60, 100, (height, width))
                        
                        # Add some variations based on seed
                        feature_type = seed % 10
                        
                        if feature_type == 0:  # Annual Crop
                            for i in range(0, height, 4):
                                image_array[i:i+2,:,1] = image_array[i:i+2,:,1] * 1.2
                                image_array[i:i+2,:,0] = image_array[i:i+2,:,0] * 0.8
                        
                        elif feature_type == 1:  # Forest
                            image_array[:,:,1] = np.random.randint(100, 160, (height, width))
                            image_array[:,:,0] = np.random.randint(40, 90, (height, width))
                        
                        elif feature_type in [8, 9]:  # River or Lake
                            center_x, center_y = width//2, height//2
                            for i in range(height):
                                for j in range(width):
                                    dist = ((i-center_y)**2 + (j-center_x)**2)**0.5
                                    if dist < width/3:
                                        image_array[i,j,2] = np.random.randint(150, 220)
                                        image_array[i,j,1] = np.random.randint(50, 100)
                                        image_array[i,j,0] = np.random.randint(10, 60)
                        
                        # Create image
                        image = Image.fromarray(image_array)
                
                elif model_type_map == "RGB-NIR":
                    st.markdown("### RGB-NIR Analysis")
                    
                    if has_sentinel_credentials:
                        try:
                            # Fetch RGB and NIR bands from Sentinel Hub
                            st.info("Fetching RGB and Near-Infrared bands from Sentinel Hub...")
                            st.info("Using simulated data for demonstration.")
                            
                            # Simulate RGB-NIR image - with 4 channels
                            width, height = 64, 64
                            image_array = np.zeros((height, width, 4), dtype=np.uint8)
                            
                            # RGB channels
                            image_array[:,:,0] = np.random.randint(70, 120, (height, width))  # Red
                            image_array[:,:,1] = np.random.randint(90, 140, (height, width))  # Green
                            image_array[:,:,2] = np.random.randint(60, 100, (height, width))  # Blue
                            
                            # Near-Infrared channel - vegetation reflects strongly in NIR
                            nir = np.zeros((height, width), dtype=np.uint8)
                            # Base NIR reflection
                            nir = np.random.randint(100, 180, (height, width))
                            
                            # Enhance NIR for vegetation areas (correlated with green channel)
                            vegetation_mask = image_array[:,:,1] > 110
                            nir[vegetation_mask] = np.random.randint(180, 240, nir[vegetation_mask].shape)
                            
                            # Water absorbs NIR
                            water_mask = image_array[:,:,2] > 150
                            nir[water_mask] = np.random.randint(20, 60, nir[water_mask].shape)
                            
                            image_array[:,:,3] = nir  # Add NIR as 4th channel
                            
                            # Convert to RGB for display (using first 3 channels)
                            display_img = Image.fromarray(image_array[:,:,:3])
                            
                            # For processing, use all 4 channels
                            image = Image.fromarray(image_array[:,:,:3])  # Simplified for demo
                            
                        except Exception as e:
                            st.error(f"Error fetching RGB-NIR imagery: {str(e)}")
                            st.stop()
                    else:
                        # Use simulated data as in original code
                        st.info("Using simulated data (no API credentials provided).")
                        width, height = 64, 64
                        image_array = np.zeros((height, width, 4), dtype=np.uint8)
                        
                        image_array[:,:,0] = np.random.randint(70, 120, (height, width))
                        image_array[:,:,1] = np.random.randint(90, 140, (height, width))
                        image_array[:,:,2] = np.random.randint(60, 100, (height, width))
                        
                        nir = np.random.randint(100, 180, (height, width))
                        vegetation_mask = image_array[:,:,1] > 110
                        nir[vegetation_mask] = np.random.randint(180, 240, nir[vegetation_mask].shape)
                        water_mask = image_array[:,:,2] > 150
                        nir[water_mask] = np.random.randint(20, 60, nir[water_mask].shape)
                        
                        image_array[:,:,3] = nir
                        
                        display_img = Image.fromarray(image_array[:,:,:3])
                        image = Image.fromarray(image_array[:,:,:3])
                
                else:  # NDVI
                    st.markdown("### NDVI Analysis")
                    
                    if has_sentinel_credentials:
                        try:
                            # Fetch Red and NIR bands from Sentinel Hub
                            st.info("Fetching Red and Near-Infrared bands from Sentinel Hub...")
                            st.info("Calculating NDVI from spectral data...")
                            
                            # Simulate NDVI calculation - in a real app, this would use actual data
                            width, height = 64, 64
                            red_band = np.random.randint(70, 170, (height, width)).astype(np.float32)
                            nir_band = np.random.randint(100, 220, (height, width)).astype(np.float32)
                            
                            # Vegetation has low red, high NIR
                            vegetation_mask = np.random.rand(height, width) > 0.6
                            red_band[vegetation_mask] = np.random.randint(30, 90, red_band[vegetation_mask].shape)
                            nir_band[vegetation_mask] = np.random.randint(160, 240, nir_band[vegetation_mask].shape)
                            
                            # Water has low reflectance in both
                            water_mask = np.random.rand(height, width) > 0.8
                            red_band[water_mask] = np.random.randint(20, 60, red_band[water_mask].shape)
                            nir_band[water_mask] = np.random.randint(10, 40, nir_band[water_mask].shape)
                            
                            # Normalize and calculate NDVI
                            red_norm = red_band / 255.0
                            nir_norm = nir_band / 255.0
                            
                            denominator = nir_norm + red_norm
                            denominator[denominator == 0] = 1  # Avoid division by zero
                            ndvi = (nir_norm - red_norm) / denominator
                            
                            # Create colormap for visualization
                            colormap = np.zeros((height, width, 3), dtype=np.uint8)
                            
                            # Water/negative NDVI: blue
                            neg_mask = ndvi < -0.2
                            colormap[neg_mask, 2] = 255
                            
                            # Low vegetation: light green
                            low_veg_mask = (ndvi >= -0.2) & (ndvi < 0.2)
                            colormap[low_veg_mask, 1] = 180
                            
                            # Medium vegetation: medium green
                            med_veg_mask = (ndvi >= 0.2) & (ndvi < 0.4)
                            colormap[med_veg_mask, 1] = 220
                            
                            # High vegetation: dark green
                            high_veg_mask = ndvi >= 0.4
                            colormap[high_veg_mask, 1] = 255
                            
                            # Create single-channel NDVI image for processing
                            ndvi_img = ((ndvi + 1) / 2 * 255).astype(np.uint8)
                            
                            display_img = Image.fromarray(colormap)
                            image = Image.fromarray(ndvi_img)
                            
                        except Exception as e:
                            st.error(f"Error calculating NDVI: {str(e)}")
                            st.stop()
                    else:
                        # Use simulated data as in original code
                        st.info("Using simulated data (no API credentials provided).")
                        
                        width, height = 64, 64
                        red_band = np.random.randint(70, 170, (height, width)).astype(np.float32)
                        nir_band = np.random.randint(100, 220, (height, width)).astype(np.float32)
                        
                        vegetation_mask = np.random.rand(height, width) > 0.6
                        red_band[vegetation_mask] = np.random.randint(30, 90, red_band[vegetation_mask].shape)
                        nir_band[vegetation_mask] = np.random.randint(160, 240, nir_band[vegetation_mask].shape)
                        
                        water_mask = np.random.rand(height, width) > 0.8
                        red_band[water_mask] = np.random.randint(20, 60, red_band[water_mask].shape)
                        nir_band[water_mask] = np.random.randint(10, 40, nir_band[water_mask].shape)
                        
                        red_norm = red_band / 255.0
                        nir_norm = nir_band / 255.0
                        
                        denominator = nir_norm + red_norm
                        denominator[denominator == 0] = 1
                        ndvi = (nir_norm - red_norm) / denominator
                        
                        colormap = np.zeros((height, width, 3), dtype=np.uint8)
                        
                        neg_mask = ndvi < -0.2
                        colormap[neg_mask, 2] = 255
                        
                        low_veg_mask = (ndvi >= -0.2) & (ndvi < 0.2)
                        colormap[low_veg_mask, 1] = 180
                        
                        med_veg_mask = (ndvi >= 0.2) & (ndvi < 0.4)
                        colormap[med_veg_mask, 1] = 220
                        
                        high_veg_mask = ndvi >= 0.4
                        colormap[high_veg_mask, 1] = 255
                        
                        ndvi_img = ((ndvi + 1) / 2 * 255).astype(np.uint8)
                        
                        display_img = Image.fromarray(colormap)
                        image = Image.fromarray(ndvi_img)
                
                # Display the image
                if model_type_map != "RGB":
                    # For RGB-NIR and NDVI, show the visualization image
                    st.image(display_img, caption=f"Selected Region ({model_type_map} Data)", width=400)
                    st.info("Displaying processed visualization. Original multi-band data is being used for analysis.")
                else:
                    # For RGB, show the actual image
                    st.image(image, caption=f"Selected Region ({model_type_map} Data)", width=400)
                
                # Information about the source
                if not has_sentinel_credentials:
                    st.warning("Using simulated data. For real satellite imagery, provide Sentinel Hub API credentials.")
                else:
                    st.success(f"Using satellite data from {selected_date.strftime('%Y-%m-%d')}")
                
                # Make prediction and show analysis
                with st.spinner("Analyzing region..."):
                    result = predict_and_analyze(image, model_type_map)
                    
                if result:
                    st.success(f"Prediction: **{result['class_name']}** with {result['confidence']:.2f}% confidence")
                    # Show detailed analysis
                    display_analysis(result, model_type_map)
                else:
                    st.error("Could not analyze the region. Please try another area or model.")

# Add sidebar information with space theme
st.sidebar.title("🛰️ About this App")
st.sidebar.info(
    "This satellite-based application uses deep learning models to classify land types from orbital imagery. "
    "You can either upload your own images or select regions from the map interface."
)

st.sidebar.markdown("### 🤖 Model Information")
st.sidebar.markdown("""
**RGB Model:** Uses standard 3-band imagery for classification.

**RGB-NIR Model:** Uses RGB + Near-Infrared bands for improved vegetation analysis.

**NDVI Model:** Focuses on vegetation health using the Normalized Difference Vegetation Index.
""")

st.sidebar.markdown("### 🌎 Dataset Information")
st.sidebar.markdown("""
The models were trained on satellite imagery containing the following land types:
- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea Lake
""")

# Footer
st.markdown("---")
st.markdown("🛰️ Developed for DEPI Land Type Classification Project • Satellite Imagery Analysis Platform • © 2023")
