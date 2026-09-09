import socket
import qrcode
import io
# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import io
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from image_module.service import load_models, predict as predict_image
from uv_module.core.audit import init_db, log_batch_results, get_audit_history
from uv_module.core.router import route_and_process

from training_scripts.train_hirda import train_hirda_model
from training_scripts.train_rauwolfia import train_rauwolfia_model
from training_scripts.train_embelia import train_embelia_model
from training_scripts.train_plant_detector import train_plant_detector_model
from training_scripts.model_manager import (
    backup_production_models, get_staging_dir, compare_and_decide,
    promote_staging, cleanup_staging, get_production_accuracy
)

ROOT = Path(__file__).resolve().parent
UV_MODELS_DIR = ROOT / 'uv_module' / 'models'
UV_DATA_DIR = ROOT / 'uv_module' / 'data'
TEMPLATES_DIR = ROOT / 'uv_module' / 'templates'
IMAGE_MODELS_DIR = ROOT / 'image_module' / 'models'

st.set_page_config(
    page_title='Integrated Plant Suite & Retraining',
    page_icon='',
    layout='wide',
    initial_sidebar_state='expanded',
)

init_db()

CUSTOM_CSS = '''
<style>
    @import url('https://fonts.googleapis.com/css2family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b1329 0%, #0f172a 50%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button[data-baseweb="button"][kind="primary"] {
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
        border: none;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
    }
    
    .stButton>button[data-baseweb="button"][kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(16, 185, 129, 0.55);
    }
</style>
'''
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner='Loading image deep learning models...')
def get_image_models():
    return load_models(IMAGE_MODELS_DIR)


def plot_confusion_matrix_fig(y_true, y_pred, labels=None, title='Confusion Matrix'):
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        ax=ax,
        cbar=True,
        annot_kws={'size': 11, 'weight': 'bold', 'color': 'white'},
        linewidths=1,
        linecolor='#1e293b'
    )
    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', color='#94a3b8', fontsize=10, fontweight='bold')
    ax.set_ylabel('Actual Label', color='#94a3b8', fontsize=10, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)

    plt.xticks(rotation=30, ha='right', color='white')
    plt.yticks(rotation=0, color='white')
    plt.tight_layout()

    return fig, cm_df


def normalize_ground_truth_label(actual_val, predicted_val, target_plant="Auto-Detect"):
    actual_str = str(actual_val).strip()
    pred_str = str(predicted_val).strip()
    
    if actual_str.lower() == pred_str.lower():
        return pred_str
        
    # Embelia Engine mapping
    if target_plant == 'Embelia' or pred_str in ['Authentic Embelia', 'Adulterant / Unknown']:
        act_low = actual_str.lower()
        if 'ribes' in act_low or 'authentic' in act_low:
            return 'Authentic Embelia'
        elif any(k in act_low for k in ['tsjeriam', 'cottom', 'cottam', 'adulterant', 'control', 'negative', 'unknown', 'foreign']):
            return 'Adulterant / Unknown'
        elif 'embelia' in act_low and 'ribes' not in act_low:
            return 'Adulterant / Unknown'

    # Rauwolfia Engine mapping
    if target_plant in ['Rauwolfia/Terminalia', 'Rauwolfia'] or any(k in pred_str.lower() for k in ['serpentina', 'tetraphylla', 'densiflora', 'foreign', 'unknown']):
        act_low = actual_str.lower()
        if 'serpentina' in act_low:
            return 'Rauwolfia serpentina'
        elif 'tetraphylla' in act_low:
            return 'Rauwolfia tetraphylla'
        elif 'densiflora' in act_low:
            return 'Rauwolfia densiflora'
        elif any(k in act_low for k in ['foreign', 'unknown', 'adulterant', 'control']):
            return 'UNKNOWN / FOREIGN'

    # Hirda Engine mapping
    for variety in ['Rohini', 'Jivanti', 'Vijaya', 'Chetaki', 'Putana', 'Amrita', 'Abhaya']:
        if actual_str.lower() == variety.lower():
            return variety
            
    return actual_str


def find_ground_truth_column(df):
    possible_cols = ['label', 'actual_label', 'species', 'variety', 'ground_truth', 'target', 'class', 'actual_class', 'type', 'group', 'name', 'sample_name', 'sample']
    for col in df.columns:
        if str(col).strip().lower() in possible_cols:
            return col
            
    # Check if the first column contains string labels rather than numeric float values
    first_col = df.columns[0]
    try:
        float(first_col)
    except ValueError:
        sample_vals = df[first_col].dropna().head(5).astype(str).tolist()
        if any(not v.replace('.','',1).isdigit() for v in sample_vals):
            return first_col
            
    return None


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


def render_mobile_qr_widget():
    local_ip = get_local_ip()
    url = f"http://{local_ip}:8501"
    
    with st.sidebar.expander("📱 Mobile & Network Access", expanded=False):
        st.markdown("**Scan to open on phone:**")
        try:
            qr = qrcode.QRCode(version=1, box_size=4, border=2)
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), use_container_width=True)
        except Exception as e:
            st.caption(f"QR Error: {e}")
            
        st.markdown("**Wi-Fi Mobile URL:**")
        st.code(url, language="text")
        st.caption("Ensure your phone is connected to the same Wi-Fi network.")


def render_sidebar():
    with st.sidebar:
        st.markdown('#  Plant Suite')
        st.markdown('**B.V. Bhide Foundation**')
        st.caption('Spectral & Computer Vision Intelligence')
        st.markdown('---')
        page = st.radio(
            'Navigation',
            [
                'UV Spectral Analysis',
                'Model Retraining',
                'Image Analysis',
                'Audit Logs',
            ],
            format_func=lambda x: {
                'UV Spectral Analysis': ' UV Spectral Analysis',
                'Model Retraining': ' Model Retraining',
                'Image Analysis': ' Image Analysis',
                'Audit Logs': ' Audit Logs'
            }[x]
        )
        st.markdown('---')
        render_mobile_qr_widget()
        st.caption('B.V. Bhide Foundation 2026')
        st.caption('')
    return page


def render_uv_page():
    st.markdown('##  UV Spectral Scan Analysis')
    st.write('Upload UV/FTIR spectral CSV files for rapid variety classification, authenticity verification, and adulteration scoring.')

    target_plant = st.selectbox(
        'Select Target Species / Mode',
        ['Auto-Detect', 'Rauwolfia/Terminalia', 'Embelia', 'Hirda'],
        help='Auto-Detect automatically identifies plant species from UV spectra before applying specialized sub-classifiers.',
    )

    template_map = {
        'Auto-Detect': ('auto_detect_template.csv', 'Auto-Detect Multi-Species Template'),
        'Rauwolfia/Terminalia': ('rauwolfia_template.csv', 'Rauwolfia Species Template'),
        'Embelia': ('embelia_template.csv', 'Embelia Authenticity Template'),
        'Hirda': ('hirda_template.csv', 'Hirda Variety Template'),
    }

    template_filename, template_label = template_map[target_plant]
    template_path = TEMPLATES_DIR / template_filename

    col1, col2 = st.columns([1, 1], gap='large')

    with col1:
        st.markdown('###  Download CSV Template')
        st.markdown(f'Use the standard template for **{target_plant}** to ensure CSV headers match the pre-processing pipeline.')
        
        if template_path.exists():
            with open(template_path, 'rb') as f:
                bytes_data = f.read()
            st.download_button(
                label=f' Download {template_label} (.csv)',
                data=bytes_data,
                file_name=template_filename,
                mime='text/csv',
                use_container_width=True,
            )
            
            with st.expander(' Format Preview & Guide'):
                t_df = pd.read_csv(template_path)
                st.markdown('**Template Preview:**')
                st.dataframe(t_df.head(3), use_container_width=True, hide_index=True)
                st.markdown('**Processing Guide:**')
                st.markdown('- **First column:** Sample ID or ground truth label (label/ariety/species).')
                st.markdown('- **Wavelength columns:** UV absorption spectrum across wavelengths (200-800nm).')
                st.markdown('- **Ground Truth:** If a label column is provided, a **Confusion Matrix** is automatically generated at the end of the run!')
        else:
            st.warning(f'Template file {template_filename} not found.')

    with col2:
        st.markdown('###  Upload & Analyze Spectral Data')
        uploaded_files = st.file_uploader(
            'Choose spectral CSV file(s)',
            type=['csv'],
            accept_multiple_files=True,
        )

        run_btn = st.button(' Run UV Analysis', type='primary', use_container_width=True)

    if uploaded_files and run_btn:
        all_results = []
        ground_truth_pairs = []

        with st.spinner('Processing UV spectral scans...'):
            for uploaded_file in uploaded_files:
                try:
                    df_raw = pd.read_csv(uploaded_file)
                    uploaded_file.seek(0)
                    gt_col = find_ground_truth_column(df_raw)
                    
                    results, modality = route_and_process(
                        uploaded_file, UV_MODELS_DIR, UV_DATA_DIR, target_plant
                    )
                    log_batch_results('integrated-user', target_plant, results)
                    
                    for idx, res in enumerate(results):
                        all_results.append(res)
                        if gt_col is not None and idx < len(df_raw):
                            actual_val = str(df_raw.iloc[idx][gt_col]).strip()
                            pred_val = str(res.get('prediction', '')).strip()
                            if actual_val and actual_val.lower() != 'nan':
                                norm_actual = normalize_ground_truth_label(actual_val, pred_val, target_plant)
                                ground_truth_pairs.append({'actual': norm_actual, 'predicted': pred_val})
                            
                    st.success(f'Successfully analyzed **{uploaded_file.name}** ({modality})')
                except Exception as error:
                    st.error(f'Error processing {uploaded_file.name}: {error}')

        if all_results:
            st.markdown('---')
            st.markdown('##  Classification Results')
            
            result_frame = pd.DataFrame(all_results)
            display_cols = [c for c in ['sample_name', 'plant_detection', 'prediction', 'adul_status', 'purity'] if c in result_frame.columns]
            st.dataframe(result_frame[display_cols], use_container_width=True, hide_index=True)

            # Confusion Matrix Section
            if ground_truth_pairs:
                st.markdown('---')
                st.markdown('##  Ground Truth Validation & Confusion Matrix')
                
                gt_df = pd.DataFrame(ground_truth_pairs)
                y_true = gt_df['actual'].tolist()
                y_pred = gt_df['predicted'].tolist()
                
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric('Validation Accuracy', f'{acc * 100:.2f}%')
                with m2:
                    st.metric('Weighted F1 Score', f'{f1 * 100:.2f}%')
                with m3:
                    st.metric('Validated Samples', f'{len(y_true)}')
                
                fig, cm_df = plot_confusion_matrix_fig(y_true, y_pred, title=f'{target_plant} Test Run Confusion Matrix')
                
                c_fig, c_tbl = st.columns([1.2, 1], gap='medium')
                with c_fig:
                    st.pyplot(fig)
                with c_tbl:
                    st.markdown('**Confusion Matrix Breakdown:**')
                    st.dataframe(cm_df, use_container_width=True)

            with st.expander(' Detailed Sample Breakdown'):
                for res in all_results:
                    st.write(f"**{res.get('sample_name', 'Sample')}** -> {res.get('prediction', 'N/A')}")
                    if 'confidence' in res:
                        st.json(res['confidence'])



def render_retraining_page():
    st.markdown('## Model Retraining Dashboard')
    st.write('Retrain models on new datasets. A failsafe system automatically backs up current models, trains to a staging area, and only promotes the new model if it outperforms the current one.')

    MODEL_KEY_MAP = {
        'Hirda Variety Classifier (Random Forest + PCA + Ratios)': 'hirda',
        'Rauwolfia Species Classifier (SNV + Imputer + PCA + SVM)': 'rauwolfia',
        'Embelia Authenticity Engine (Fingerprint Vector + Threshold)': 'embelia',
        'Tier-1 Plant Auto-Detector (Multi-Species UV Model)': 'plant_detector',
    }

    model_choice = st.selectbox(
        'Select Model to Retrain',
        list(MODEL_KEY_MAP.keys()),
    )
    model_key = MODEL_KEY_MAP[model_choice]

    col1, col2 = st.columns([1, 1], gap='large')

    with col1:
        st.markdown('### Data Source Selection')
        data_source = st.radio(
            'Choose Training Data',
            ['Use Pre-packaged Suite Dataset', 'Upload New Custom CSV Dataset'],
        )

        custom_file = None
        if data_source == 'Upload New Custom CSV Dataset':
            custom_file = st.file_uploader('Upload Custom Training CSV', type=['csv'])
            if custom_file:
                st.success(f'Uploaded {custom_file.name} ready for retraining.')

        retrain_btn = st.button('Execute Model Retraining', type='primary', use_container_width=True)

    with col2:
        st.markdown('### Retraining Details')
        if model_key == 'hirda':
            st.info('**Hirda:** Phytochemical ratio features + PCA + Random Forest across 7 varieties.')
        elif model_key == 'rauwolfia':
            st.info('**Rauwolfia:** SNV baseline correction + mean imputation + PCA + SVM classification.')
        elif model_key == 'embelia':
            st.info('**Embelia:** Mean reference fingerprint + 95th percentile outlier threshold.')
        else:
            st.info('**Plant Detector:** Multi-species UV Random Forest auto-router.')

        # Show current production accuracy
        prod_acc = get_production_accuracy(model_key, UV_MODELS_DIR, UV_DATA_DIR)
        if prod_acc > 0:
            st.metric('Current Production Accuracy', f'{prod_acc:.2f}%')
        else:
            st.caption('No stored production accuracy found (first training run).')

    if retrain_btn:
        st.markdown('---')
        st.markdown('### Retraining with Failsafe Protection')
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Backup
            status_text.text('Step 1/4: Backing up current production models...')
            progress_bar.progress(10)
            backup_production_models(model_key, UV_MODELS_DIR, UV_DATA_DIR)

            # Step 2: Prepare dataset
            status_text.text('Step 2/4: Preparing dataset...')
            progress_bar.progress(20)

            if data_source == 'Upload New Custom CSV Dataset' and custom_file is not None:
                temp_path = UV_DATA_DIR / f'custom_train_{custom_file.name}'
                with open(temp_path, 'wb') as f:
                    f.write(custom_file.getbuffer())
                dataset_path = temp_path
            else:
                if model_key == 'hirda':
                    dataset_path = UV_DATA_DIR / 'Terminalia_UV_new.csv'
                elif model_key == 'rauwolfia':
                    dataset_path = UV_DATA_DIR / 'Rauwolfia_UV.csv'
                elif model_key == 'embelia':
                    dataset_path = UV_DATA_DIR / 'Embelia_UV_new.csv'
                else:
                    dataset_path = UV_DATA_DIR

            # Step 3: Train to STAGING directory
            status_text.text('Step 3/4: Training new model to staging area...')
            progress_bar.progress(40)
            staging = get_staging_dir(model_key, UV_MODELS_DIR)

            if model_key == 'hirda':
                res = train_hirda_model(dataset_path, UV_MODELS_DIR, staging_dir=staging)
            elif model_key == 'rauwolfia':
                res = train_rauwolfia_model(dataset_path, UV_MODELS_DIR, staging_dir=staging)
            elif model_key == 'embelia':
                res = train_embelia_model(dataset_path, UV_DATA_DIR, staging_dir=staging)
            else:
                res = train_plant_detector_model(UV_DATA_DIR, UV_MODELS_DIR, staging_dir=staging)

            progress_bar.progress(80)

            # Step 4: Compare and decide
            status_text.text('Step 4/4: Comparing new model vs production...')
            new_acc = res.get('cv_accuracy', res.get('accuracy', 0.0))
            decision = compare_and_decide(model_key, new_acc, UV_MODELS_DIR, UV_DATA_DIR)
            progress_bar.progress(100)

            # Display metrics comparison
            st.markdown('---')
            st.markdown('## Retraining Results & Comparison')

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric('Production Accuracy', f"{decision['old_accuracy']:.2f}%")
            with m2:
                st.metric('New Model Accuracy', f"{decision['new_accuracy']:.2f}%")
            with m3:
                delta_val = decision['delta']
                st.metric('Accuracy Delta', f"{delta_val:+.2f}%")
            with m4:
                st.metric('Classes Trained', f"{len(res['classes'])}")

            # Confusion Matrix
            cm_matrix = res['confusion_matrix']
            fig_cm, ax_cm = plt.subplots(figsize=(6.5, 4.5))
            fig_cm.patch.set_facecolor('#0f172a')
            ax_cm.set_facecolor('#0f172a')
            sns.heatmap(
                pd.DataFrame(cm_matrix, index=res['classes'], columns=res['classes']),
                annot=True,
                fmt='d',
                cmap='Greens' if decision['improved'] else 'Reds',
                ax=ax_cm,
                cbar=True,
                annot_kws={'size': 11, 'weight': 'bold', 'color': 'white'},
                linewidths=1,
                linecolor='#1e293b'
            )
            ax_cm.set_title('New Model Confusion Matrix', color='white', fontsize=13, fontweight='bold', pad=12)
            ax_cm.set_xlabel('Predicted Class', color='#94a3b8', fontsize=10, fontweight='bold')
            ax_cm.set_ylabel('True Class', color='#94a3b8', fontsize=10, fontweight='bold')
            ax_cm.tick_params(colors='white', labelsize=9)
            plt.xticks(rotation=30, ha='right', color='white')
            plt.yticks(rotation=0, color='white')
            plt.tight_layout()

            c_fig, c_report = st.columns([1.2, 1], gap='medium')
            with c_fig:
                st.pyplot(fig_cm)
            with c_report:
                st.markdown('**Classification Report:**')
                report_df = pd.DataFrame(res['report']).transpose()
                st.dataframe(report_df.style.format(precision=3), use_container_width=True)

            # Decision banner
            st.markdown('---')
            if decision['improved']:
                # Auto-promote
                promote_staging(model_key, UV_MODELS_DIR, UV_DATA_DIR)
                cleanup_staging(model_key, UV_MODELS_DIR)
                status_text.text('New model promoted to production!')
                st.success(
                    f"NEW MODEL PROMOTED! Accuracy improved by {decision['delta']:+.2f}% "
                    f"({decision['old_accuracy']:.2f}% -> {decision['new_accuracy']:.2f}%). "
                    f"Production models updated. Backup saved in models_backup/{model_key}/."
                )
            else:
                # Reject - keep staging for potential manual override
                status_text.text('New model performs worse - production models preserved.')
                st.error(
                    f"NEW MODEL REJECTED (Failsafe). Accuracy dropped by {abs(decision['delta']):.2f}% "
                    f"({decision['old_accuracy']:.2f}% -> {decision['new_accuracy']:.2f}%). "
                    f"Production models are UNCHANGED."
                )
                st.warning('The new model performed worse than the current production model. Your original models are safe and untouched.')

                # Offer manual override
                st.markdown('---')
                st.markdown('### Manual Override')
                st.caption('If you still want to use the new model despite lower accuracy, you can force-promote it below.')
                
                force_key = f'force_promote_{model_key}'
                if st.button('Force Promote Anyway (Override Failsafe)', type='secondary', key=force_key):
                    promote_staging(model_key, UV_MODELS_DIR, UV_DATA_DIR)
                    cleanup_staging(model_key, UV_MODELS_DIR)
                    st.success(
                        f"FORCE PROMOTED. New model ({decision['new_accuracy']:.2f}%) is now in production. "
                        f"Previous model ({decision['old_accuracy']:.2f}%) saved in models_backup/{model_key}/."
                    )

        except Exception as error:
            progress_bar.progress(100)
            status_text.text('Retraining failed.')
            st.error(f'Retraining failed: {error}')


def render_image_page():
    st.markdown('##  Deep Learning Image Analysis')
    st.write('Upload plant species images or take a live photo for MobileNetV2 image classification.')

    col1, col2 = st.columns([1, 1], gap='large')
    with col1:
        upload = st.file_uploader('Upload plant image', type=['jpg', 'jpeg', 'png', 'heic', 'heif'])
        camera = st.camera_input('Or capture photo with camera')
        image_source = camera if camera is not None else upload

    with col2:
        if image_source is not None:
            image_bytes = image_source.getvalue()
            st.image(image_bytes, caption='Selected Image', width=340)
            if st.button(' Run Image Analysis', type='primary', use_container_width=True):
                try:
                    result = predict_image(image_bytes, get_image_models())
                    st.success(f"Detected Plant: **{result['plant_label']}**")
                    st.metric('Plant Detection Confidence', f"{result['plant_confidence']:.2f}%")
                    
                    top = result['top_prediction']
                    st.markdown(f"### Prediction: {top['name']}")
                    st.progress(float(top['confidence']) / 100.0)
                    st.write(top['description'])
                    st.markdown(f"**Traditional / Pharmacological Uses:** {top['uses']}")

                    if result['alternatives']:
                        st.markdown('**Alternative Predictions:**')
                        st.dataframe(
                            pd.DataFrame(result['alternatives'])[['name', 'confidence']],
                            use_container_width=True,
                            hide_index=True,
                        )
                except Exception as error:
                    st.error(f'Image analysis failed: {error}')
        else:
            st.info('Upload an image or take a photo to begin.')


def render_audit_page():
    st.markdown('##  Historical Audit Logs & Database')
    st.write('Review past UV scan predictions, adulteration records, and audit logs stored in the SQLite database.')

    try:
        logs = get_audit_history()
        if logs:
            df_logs = pd.DataFrame(logs)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric('Total Scans Logged', len(df_logs))
            with m2:
                adul_count = len(df_logs[df_logs['status'].str.contains('Adulterated|Outlier', na=False)])
                st.metric('Adulterated / Outlier Scans', adul_count)
            with m3:
                auth_count = len(df_logs[df_logs['status'].str.contains('Authentic', na=False)])
                st.metric('Authentic Scans', auth_count)

            st.markdown('---')
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
        else:
            st.info('No audit logs recorded in database yet.')
    except Exception as e:
        st.error(f'Could not fetch audit logs: {e}')


def main():
    page = render_sidebar()
    if page == 'UV Spectral Analysis':
        render_uv_page()
    elif page == 'Model Retraining':
        render_retraining_page()
    elif page == 'Image Analysis':
        render_image_page()
    elif page == 'Audit Logs':
        render_audit_page()

if __name__ == '__main__':
    main()