# -*- coding: utf-8 -*-
with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Update import line
old_import = "from image_module.service import load_models, predict as predict_image"
new_import = "from image_module.service import load_models, predict as predict_image, is_tensorflow_available"

if old_import in text:
    text = text.replace(old_import, new_import)

# Update get_image_models
old_get_models = """@st.cache_resource(show_spinner='Loading image deep learning models...')
def get_image_models():
    return load_models(IMAGE_MODELS_DIR)"""

new_get_models = """@st.cache_resource(show_spinner='Loading image deep learning models...')
def get_image_models():
    try:
        return load_models(IMAGE_MODELS_DIR)
    except Exception:
        return None"""

if old_get_models in text:
    text = text.replace(old_get_models, new_get_models)

# Update render_image_page header check
old_render_hdr = """def render_image_page():
    st.markdown('##  Deep Learning Image Analysis')
    st.write('Upload plant species images or take a live photo for MobileNetV2 image classification.')"""

new_render_hdr = """def render_image_page():
    st.markdown('##  Deep Learning Image Analysis')
    st.write('Upload plant species images or take a live photo for MobileNetV2 image classification.')

    if not is_tensorflow_available():
        st.warning('⚠️ **TensorFlow module is not installed on this server environment.**')
        st.info('To enable deep learning image classification on Streamlit Cloud, add 	ensorflow-cpu>=2.12.0 to your equirements.txt.')
        return

    models = get_image_models()
    if models is None:
        st.error('⚠️ **Could not load Keras image classification models.** Ensure .keras model files exist in image_module/models.')
        return"""

if old_render_hdr in text:
    text = text.replace(old_render_hdr, new_render_hdr)

# Update predict call in render_image_page to pass models variable
old_pred_call = "result = predict_image(image_bytes, get_image_models())"
new_pred_call = "result = predict_image(image_bytes, models)"

if old_pred_call in text:
    text = text.replace(old_pred_call, new_pred_call)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('app.py updated with TensorFlow fallback handling!')
