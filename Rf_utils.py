# RF_ultis.py
#### FUNCTION: Define a function to update the filename .png to _Annotation.png ####
def update_filename(filename):
    return filename.replace(".png", "_Annotation.png")

def extract_id_from_filename(filename):
    parts = filename.split('_')
    if len(parts) > 1:
        id_str = parts[0]
        id_str = id_str.lstrip('0')  # Remove leading zeros
        
        if len(parts[1]) > 1 and parts[1][0].isdigit() and parts[1][1].isalpha():
            id_str += "_" + parts[1][0]
        
        return id_str
    
    return None

def map_id_to_filename(id):
    id_str = str(id).zfill(3)  # Pad ID with leading zeros
    filename = f"{id_str}_HC.png"
    return filename
    

##### function get_pixe_count get_pixel_count(img_data_path, filename) ###
## example use: 
#       train_pixel_sz_HC_df['pixel_count'] = train_pixel_sz_HC_df['anno_filename'].apply(lambda x: get_pixel_count(train_img_data_path, x))


def get_pixel_count(img_data_path, filename):
    file_path = os.path.join(img_data_path, filename)
    if os.path.exists(file_path):
        label = Image.open(file_path)
        label = np.array(label)
        label = tf.convert_to_tensor(label)
        label = tf.cast(label, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        pixel_count = np.sum(label)
    else:
        print("No such file: " + filename)
        pixel_count = np.nan

    return pixel_count

def extract_id_from_filename(filename):
    parts = filename.split('_')
    if len(parts) > 1:
        id_str = parts[0]
        id_str = id_str.lstrip('0')  # Remove leading zeros
        
        if len(parts[1]) > 1 and parts[1][0].isdigit() and parts[1][1].isalpha():
            id_str += "_" + parts[1][0]
        
        return id_str
    
    return None

def map_id_to_filename(id):
    id_str = str(id).zfill(3)  # Pad ID with leading zeros
    filename = f"{id_str}_HC.png"
    return filename

# Load the RF model
#rf_model = joblib.load("path/to/rf_model.joblib")

def get_pixel_count_gradio(input_img):
    # Add your implementation to calculate the pixel count from the input image
    pixel_count = 0  # Replace with your own code
    return pixel_count

def predict_head_circumference(input_img, pixel_size):
    ## need rf_model ####
    # Use the pixel size and other features to make a prediction
    feature_names = ['pixel_count', 'pixel size(mm)']  # Add other relevant features here
    pixel_count = get_pixel_count_gradio(input_img)
    
    prediction = rf_model.predict([[pixel_count, pixel_size]])[0]
    
    return prediction

### read csv
# RF_output_file = "Output/everything-val_pred_HC_pixel_sz.csv"
#RF_Output_df = pd.read_csv(RF_output_file, sep=',')

## function get id give head circum
def get_headcircum_from_id(id):
    ## need RF output csv / df ##
    row = RF_Output_df[RF_Output_df['id'] == id]
    headcircum = row['head circumference (mm)'].values[0] if not row.empty else None
    return headcircum

## function get filename give head circum
def get_headcircum_from_filename(filename):
    id = extract_id_from_filename(filename)
    print(id)
    row = RF_Output_df[RF_Output_df['id'] == id]
    headcircum = row['hhead circumference (mm)'].values[0] if not row.empty else None
    return headcircum


## rf_model_on_full_X   model name 
### val_pixel_sz_HC_df[selected_columns].to_csv(r'Output/demo_input.csv', index=False)
