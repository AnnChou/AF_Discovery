# RF_ultis.py
#### FUNCTION: Define a function to update the filename .png to _Annotation.png ####
def update_filename(filename):
    return filename.replace(".png", "_Annotation.png")
    

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

# Load the RF model
#rf_model = joblib.load("path/to/rf_model.joblib")

def get_pixel_count_gradio(input_img):
    # Add your implementation to calculate the pixel count from the input image
    pixel_count = 0  # Replace with your own code
    return pixel_count

def predict_head_circumference(input_img, pixel_size):
    # Use the pixel size and other features to make a prediction
    feature_names = ['pixel_count', 'pixel size(mm)']  # Add other relevant features here
    pixel_count = get_pixel_count_gradio(input_img)
    
    prediction = rf_model.predict([[pixel_count, pixel_size]])[0]
    
    return prediction

## rf_model_on_full_X   model name 
### val_pixel_sz_HC_df[selected_columns].to_csv(r'Output/demo_input.csv', index=False)