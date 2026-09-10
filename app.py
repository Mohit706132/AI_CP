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

from image_module.service import load_models, predict as predict_image, is_tensorflow_available
from uv_module.core.audit import init_db, log_batch_results, log_image_result, get_audit_history
import importlib
import uv_module.core.auth
importlib.reload(uv_module.core.auth)
from uv_module.core.auth import (
    verify_login, register_user, get_pending_users, approve_user,
    reject_user, update_user_role, get_all_users, delete_user
)
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

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1b2a 0%, #080d16 90%);
        color: #f1f5f9;
    }
    
    /* Header & Branding */
    .brand-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .brand-sub {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Glassmorphic Cards */
    .app-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
    }
    
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .brand-title {
            font-size: 1.6rem !important;
        }
        .app-card {
            padding: 14px !important;
        }
        .stMetric {
            padding: 10px !important;
        }
    }
    
    /* Custom Metric Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }
    
    /* Buttons */
    .stButton>button[data-baseweb="button"][kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 0.55rem 1.2rem;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.35);
        transition: all 0.2s ease;
    }
    
    .stButton>button[data-baseweb="button"][kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(16, 185, 129, 0.5);
    }
    
    .stButton>button[data-baseweb="button"][kind="secondary"] {
        background: rgba(30, 41, 59, 0.6);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    
    .stButton>button[data-baseweb="button"][kind="secondary"]:hover {
        background: rgba(51, 65, 85, 0.8);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Badges */
    .badge-admin {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-user {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner='Loading image deep learning models...')
def get_image_models():
    try:
        return load_models(IMAGE_MODELS_DIR)
    except Exception:
        return None


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


def render_auth_page():
    st.markdown('<div style="text-align: center; margin-top: 20px; margin-bottom: 25px;">'
                '<h1 style="color: #60a5fa; margin-bottom: 0;">🌿 Integrated Plant Authentication Suite</h1>'
                '<p style="color: #94a3b8; font-size: 1.1rem;">Botanical Intelligence, Spectral Verification & Deep Learning</p>'
                '</div>', unsafe_allow_html=True)

    c_left, c_mid, c_right = st.columns([1, 1.4, 1])
    with c_mid:
        auth_mode = st.radio("Access Portal", ["🔑 Login", "📝 Register New Account"], horizontal=True, label_visibility="collapsed")
        
        if auth_mode == "🔑 Login":
            st.markdown("### User Sign In")
            st.write("Enter your credentials to access the Plant Authentication Suite.")
            
            with st.form("login_form"):
                username = st.text_input("Username").strip()
                password = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                
                if submit_login:
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    else:
                        ok, msg, user_dict = verify_login(username, password)
                        if ok and user_dict:
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = user_dict
                            st.success(f"Welcome back, {user_dict.get('full_name', username)}!")
                            st.rerun()
                        else:
                            st.error(msg)
                            
        else:
            st.markdown("### Request System Access")
            st.write("Submit your registration. An Administrator will review and approve your account.")
            
            with st.form("register_form"):
                reg_name = st.text_input("Full Name").strip()
                reg_user = st.text_input("Desired Username").strip()
                reg_pass = st.text_input("Password", type="password")
                reg_pass2 = st.text_input("Confirm Password", type="password")
                submit_reg = st.form_submit_button("Submit Access Request", type="primary", use_container_width=True)
                
                if submit_reg:
                    if not reg_user or not reg_pass:
                        st.error("Username and password are required.")
                    elif reg_pass != reg_pass2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = register_user(reg_user, reg_pass, reg_name)
                        if ok:
                            st.success(msg)
                            st.info("Once approved by the Administrator, you will be able to log in.")
                        else:
                            st.error(msg)



def render_sidebar():
    user = st.session_state.get("user", {})
    role = user.get("role", "user")
    username = user.get("username", "User")
    full_name = user.get("full_name", username)

    with st.sidebar:
        st.markdown("<div style='display: flex; align-items: center; gap: 10px; margin-bottom: 5px;'>"
                    "<span style='font-size: 1.8rem;'>🌿</span>"
                    "<div><h2 style='margin:0; font-size: 1.3rem; font-weight: 700; color: #f8fafc;'>Plant Suite</h2>"
                    "<span style='color: #64748b; font-size: 0.75rem; font-weight: 500;'>B.V. Bhide Foundation</span></div>"
                    "</div>", unsafe_allow_html=True)
        st.caption("Spectral AI & Computer Vision")
        st.markdown("---")
        
        # User profile badge
        st.markdown(f"**{full_name}** (`@{username}`)")
        if role == "admin":
            st.markdown("<span class='badge-admin'>👑 Administrator</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-user'>🔬 Research User</span>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.rerun()

        st.markdown("---")

        # RBAC Navigation
        if role == "admin":
            nav_options = [
                "UV Spectral Analysis",
                "Image Analysis",
                "Model Retraining",
                "Audit Logs",
                "User Management & Approvals"
            ]
            format_dict = {
                "UV Spectral Analysis": "🔬 UV Spectral Scan",
                "Image Analysis": "📸 Plant Image Scan",
                "Model Retraining": "⚙️ Model Retraining",
                "Audit Logs": "📜 Security Audit Logs",
                "User Management & Approvals": "👥 User Management"
            }
        else:
            nav_options = [
                "UV Spectral Analysis",
                "Image Analysis"
            ]
            format_dict = {
                "UV Spectral Analysis": "🔬 UV Spectral Scan",
                "Image Analysis": "📸 Plant Image Scan"
            }

        page = st.radio(
            "Navigation Menu",
            nav_options,
            format_func=lambda x: format_dict[x]
        )
        st.markdown("---")
        st.caption("B.V. Bhide Foundation © 2026")
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
                    active_user = st.session_state.get("user", {}).get("username", "user")
                    log_batch_results(active_user, target_plant, results)
                    
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
    st.markdown('## Deep Learning Image Analysis')
    st.write('Upload plant species images or take a live photo for MobileNetV2 image classification.')

    if not is_tensorflow_available():
        st.warning('TensorFlow module is not installed on this server environment.')
        st.info('To enable deep learning image classification on Streamlit Cloud, ensure tensorflow-cpu is in requirements.txt.')
        return

    models = get_image_models()
    if models is None:
        st.error('Could not load Keras image classification models.')
        return

    col1, col2 = st.columns([1, 1], gap='large')
    with col1:
        upload = st.file_uploader('Upload plant image', type=['jpg', 'jpeg', 'png', 'heic', 'heif'])
        camera = st.camera_input('Or capture photo with camera')
        image_source = camera if camera is not None else upload

    with col2:
        if image_source is not None:
            image_bytes = image_source.getvalue()
            st.image(image_bytes, caption='Selected Image', width=340)
            if st.button('Run Image Analysis', type='primary', use_container_width=True):
                try:
                    result = predict_image(image_bytes, models)
                    active_user = st.session_state.get("user", {}).get("username", "user")
                    log_image_result(active_user, image_name, result)
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


def render_user_management_page():
    active_admin = st.session_state.get("user", {}).get("username", "admin")
    st.markdown("<div class='brand-title'>👥 User Access & Role Administration</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-sub'>Approve new user registrations, promote operators to Administrator, or revoke access.</div>", unsafe_allow_html=True)

    all_users = get_all_users()
    pending_users = get_pending_users()
    approved_users = [u for u in all_users if u["status"] == "approved"]
    admin_users = [u for u in all_users if u["role"] == "admin"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Users", len(all_users))
    with m2:
        st.metric("Pending Approvals", len(pending_users), delta=f"{len(pending_users)} pending" if pending_users else None)
    with m3:
        st.metric("Approved Users", len(approved_users))
    with m4:
        st.metric("Active Admins", len(admin_users))

    st.markdown("---")
    
    tab_pend, tab_roles, tab_all = st.tabs(["⏳ Pending Approvals", "👑 Role Assignment & Promotion", "📋 All User Accounts"])

    with tab_pend:
        st.markdown("### ⏳ Pending Registration Requests")
        if pending_users:
            for u in pending_users:
                with st.container():
                    c_info, c_app, c_rej = st.columns([3, 1, 1], gap="small")
                    with c_info:
                        st.markdown(f"**{u.get('full_name', u['username'])}** (`@{u['username']}`)")
                        st.caption(f"Registered: `{u['created_at']}` | Requested Role: `{u['role']}`")
                    with c_app:
                        if st.button("✅ Approve", key=f"app_{u['username']}", type="primary", use_container_width=True):
                            approve_user(u["username"], active_admin)
                            st.success(f"Approved @{u['username']}")
                            st.rerun()
                    with c_rej:
                        if st.button("❌ Reject", key=f"rej_{u['username']}", use_container_width=True):
                            reject_user(u["username"])
                            st.warning(f"Rejected @{u['username']}")
                            st.rerun()
                    st.markdown("<hr style='margin: 8px 0; border: 0.5px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        else:
            st.info("✅ No pending user registration requests. All accounts have been reviewed.")

    with tab_roles:
        st.markdown("### 👑 Promote User to Administrator")
        st.write("Grant administrative privileges (Model Retraining, Audit Logs, User Approvals) to any approved user.")
        
        non_admin_users = [u for u in all_users if u["status"] == "approved" and u["username"] != "admin"]
        if non_admin_users:
            c_select, c_role, c_btn = st.columns([2, 1.5, 1], gap="medium")
            with c_select:
                user_options = {u["username"]: f"{u.get('full_name', u['username'])} (@{u['username']}) - Currently: {u['role'].upper()}" for u in non_admin_users}
                target_user = st.selectbox("Select User Account", list(user_options.keys()), format_func=lambda x: user_options[x])
            with c_role:
                new_role = st.selectbox("Assign Role", ["admin", "user"], format_func=lambda x: "👑 Administrator (Full Control)" if x == "admin" else "🔬 Research User (Scan Only)")
            with c_btn:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("Update Role", type="primary", use_container_width=True):
                    update_user_role(target_user, new_role)
                    st.success(f"Successfully updated @{target_user} to {new_role.upper()}!")
                    st.rerun()
        else:
            st.info("No candidate users available for role modification.")

    with tab_all:
        st.markdown("### 📋 All Registered Users Directory")
        if all_users:
            df_users = pd.DataFrame(all_users)
            display_cols = [c for c in ["id", "username", "full_name", "role", "status", "created_at", "approved_by", "last_login"] if c in df_users.columns]
            st.dataframe(df_users[display_cols], use_container_width=True, hide_index=True)


def main():
    if not st.session_state.get("authenticated", False):
        render_auth_page()
    else:
        page = render_sidebar()
        if page == "UV Spectral Analysis":
            render_uv_page()
        elif page == "Model Retraining":
            render_retraining_page()
        elif page == "Image Analysis":
            render_image_page()
        elif page == "Audit Logs":
            render_audit_page()
        elif page == "User Management & Approvals":
            render_user_management_page()

if __name__ == '__main__':
    main()
