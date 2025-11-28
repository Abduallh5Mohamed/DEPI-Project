import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from styles import get_custom_css, card_component

# Configuration
st.set_page_config(
    page_title="🎓 Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load data from CSV file with proper column mapping for the dashboard."""
    csv_path = os.path.join('data', 'raw_student_data.csv')
    
    if not os.path.exists(csv_path):
        st.error('❌ Data file not found: data/raw_student_data.csv')
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        
        # Map CSV columns to expected dashboard columns
        # CSV: student_id, subject_id, teacher_id, exam_1, exam_2, exam_3, attendance, final_score, date
        # Expected: student_name, subject_name, teacher_name, exam_1, exam_2, exam_3, total_attendance, total_mark, class, gender
        
        df = df.rename(columns={
            'student_id': 'student_name',
            'subject_id': 'subject_name', 
            'teacher_id': 'teacher_name',
            'attendance': 'total_attendance',
            'final_score': 'total_mark',
            'date': 'exam_date'
        })
        
        # Clean up subject names for better display (e.g., SUB_BIO_910 → Bio 910)
        if 'subject_name' in df.columns:
            df['subject_name'] = df['subject_name'].apply(
                lambda x: str(x).replace('SUB_', '').replace('_', ' ').title()[:20] if pd.notna(x) else 'Unknown'
            )
        
        # Clean up student names (e.g., STU_ABC_123 → Student ABC123)
        if 'student_name' in df.columns:
            df['student_name'] = df['student_name'].apply(
                lambda x: 'Student ' + str(x).replace('STU_', '').replace('_', '')[:10] if pd.notna(x) else 'Unknown'
            )
        
        # Clean up teacher names
        if 'teacher_name' in df.columns:
            df['teacher_name'] = df['teacher_name'].apply(
                lambda x: 'Teacher ' + str(x).replace('TCH_', '').replace('_', '')[:10] if pd.notna(x) else 'Unknown'
            )
        
        # Add missing columns with default/generated values
        if 'class' not in df.columns:
            # Generate class from subject or random assignment
            np.random.seed(42)
            classes = ['Class A', 'Class B', 'Class C', 'Class D']
            df['class'] = np.random.choice(classes, size=len(df))
        
        if 'gender' not in df.columns:
            # Assign random gender for demo purposes (consistent with seed)
            np.random.seed(42)
            df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
        
        if 'score_id' not in df.columns:
            df['score_id'] = range(1, len(df) + 1)
        
        # Ensure numeric columns are numeric
        for col in ['exam_1', 'exam_2', 'exam_3', 'total_attendance', 'total_mark']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert exam_date to datetime if present
        if 'exam_date' in df.columns:
            df['exam_date'] = pd.to_datetime(df['exam_date'], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f'❌ Error loading data: {str(e)}')
        return pd.DataFrame()


@st.cache_resource
def load_models():
    """Load ML models if available, return empty dict with warning if not found."""
    models = {}
    models_available = True
    
    try:
        reg_path = 'models/score_predictor.pkl'
        clf_path = 'models/performance_classifier.pkl'
        
        if os.path.exists(reg_path):
            with open(reg_path, 'rb') as f:
                models['reg'] = pickle.load(f)
        else:
            models_available = False
            
        if os.path.exists(clf_path):
            with open(clf_path, 'rb') as f:
                models['clf'] = pickle.load(f)
        else:
            models_available = False
            
    except Exception:
        models_available = False
    
    return models


# Load data and models
df = load_data()
models = load_models()

# Sidebar Navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview", 
    "📈 Advanced Insights", 
    "🧬 Statistical Analysis", 
    "🔮 Predictive Analytics",
    "🎯 Risk Analysis",
    "🏆 Performance Benchmarking",
    "🤖 ML Insights", 
    "📄 Raw Data"
])
page = page.split(" ", 1)[1]  # Remove emoji from page name

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filters")

# Validate data is loaded
if df is None or df.empty:
    st.error("❌ No data available. Please ensure data/raw_student_data.csv exists.")
    st.stop()

# Check for required columns
required_cols = {'subject_name', 'class', 'total_mark', 'total_attendance', 'student_name'}
available_cols = set(df.columns)
missing_cols = required_cols - available_cols

if missing_cols:
    st.error(f"❌ Missing required columns: {', '.join(sorted(missing_cols))}")
    st.info(f"📋 Available columns: {', '.join(sorted(df.columns))}")
    st.stop()

# Safe filters
selected_subject = st.sidebar.multiselect(
    "📚 Select Subject", 
    sorted(df['subject_name'].unique()), 
    default=sorted(df['subject_name'].unique())[:5] if len(df['subject_name'].unique()) > 5 else sorted(df['subject_name'].unique())
)
selected_class = st.sidebar.multiselect(
    "🎓 Select Class", 
    sorted(df['class'].unique()), 
    default=sorted(df['class'].unique())
)

# Filter Data
if selected_subject and selected_class:
    filtered_df = df[df['subject_name'].isin(selected_subject) & df['class'].isin(selected_class)]
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")
    st.stop()


# ============== PAGE: OVERVIEW ==============
if page == "Overview":
    st.title("🎓 Student Performance Dashboard")
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    avg_score = filtered_df['total_mark'].mean()
    avg_attendance = filtered_df['total_attendance'].mean()
    total_students = filtered_df['student_name'].nunique()
    
    if not filtered_df.empty and 'total_mark' in filtered_df.columns:
        top_idx = filtered_df['total_mark'].idxmax()
        top_performer = filtered_df.loc[top_idx, 'student_name'] if pd.notna(top_idx) else "N/A"
    else:
        top_performer = "N/A"
    
    with col1:
        st.markdown(card_component("Average Score", f"{avg_score:.1f}", "Across all subjects"), unsafe_allow_html=True)
    with col2:
        st.markdown(card_component("Avg Attendance", f"{avg_attendance:.1f}/10", "Lectures attended"), unsafe_allow_html=True)
    with col3:
        st.markdown(card_component("Total Students", f"{total_students}", "Active students"), unsafe_allow_html=True)
    with col4:
        display_name = str(top_performer).split()[1] if len(str(top_performer).split()) > 1 else str(top_performer)[:10]
        st.markdown(card_component("Top Performer", display_name, "Highest score"), unsafe_allow_html=True)

    st.markdown("### 📊 Performance Trends & Analytics")
    
    # Chart: Average Score by Subject
    subject_avg = filtered_df.groupby('subject_name')['total_mark'].mean().reset_index()
    fig_bar = px.bar(
        subject_avg,
        x='subject_name',
        y='total_mark',
        title="📊 Average Score by Subject"
    )
    fig_bar.update_traces(marker_color='#06b6d4', marker_line_color='white', marker_line_width=2)
    fig_bar.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0f172a', family='Inter', size=13),
        title_font_size=20,
        xaxis=dict(showgrid=False, title='Subject'),
        yaxis=dict(gridcolor='#f1f5f9', title='Average Score'),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Two column charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(filtered_df, x='total_mark', nbins=20, title="📈 Score Distribution")
        fig_hist.update_traces(marker_color='#06b6d4')
        fig_hist.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter'),
            xaxis=dict(title='Score'),
            yaxis=dict(title='Count', gridcolor='#f1f5f9'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_scatter = px.scatter(
            filtered_df, 
            x='total_attendance', 
            y='total_mark',
            color='subject_name',
            title="📉 Attendance vs Score",
            opacity=0.7
        )
        fig_scatter.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter'),
            xaxis=dict(title='Attendance', gridcolor='#f1f5f9'),
            yaxis=dict(title='Score', gridcolor='#f1f5f9'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ============== PAGE: ADVANCED INSIGHTS ==============
elif page == "Advanced Insights":
    st.title("📈 Advanced Analytics & Insights")
    
    # Subject Performance Analysis
    st.markdown("### 🎯 Subject Performance Analysis")
    
    subject_stats = filtered_df.groupby('subject_name').agg({
        'total_mark': ['mean', 'std', 'min', 'max', 'count'],
        'total_attendance': 'mean'
    }).round(2)
    subject_stats.columns = ['Avg Score', 'Std Dev', 'Min Score', 'Max Score', 'Count', 'Avg Attendance']
    subject_stats = subject_stats.reset_index()
    
    st.dataframe(subject_stats, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot by subject
        fig_box = px.box(filtered_df, x='subject_name', y='total_mark', title="📊 Score Distribution by Subject")
        fig_box.update_traces(marker_color='#06b6d4')
        fig_box.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter'),
            xaxis=dict(title='Subject'),
            yaxis=dict(title='Score', gridcolor='#f1f5f9'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Attendance by class
        class_stats = filtered_df.groupby('class')['total_attendance'].mean().reset_index()
        fig_att = px.bar(class_stats, x='class', y='total_attendance', title="📅 Avg Attendance by Class")
        fig_att.update_traces(marker_color='#14b8a6')
        fig_att.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter'),
            xaxis=dict(title='Class'),
            yaxis=dict(title='Avg Attendance', gridcolor='#f1f5f9'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_att, use_container_width=True)
    
    # Performance Heatmap
    st.markdown("### 🌡️ Subject Performance Heatmap")
    
    if 'exam_date' in filtered_df.columns and filtered_df['exam_date'].notna().any():
        filtered_df['month'] = filtered_df['exam_date'].dt.to_period('M').astype(str)
        pivot_data = filtered_df.groupby(['subject_name', 'month'])['total_mark'].mean().reset_index()
        pivot_table = pivot_data.pivot(index='subject_name', columns='month', values='total_mark')
        
        if not pivot_table.empty:
            fig_heat = px.imshow(
                pivot_table,
                labels=dict(x="Month", y="Subject", color="Avg Score"),
                color_continuous_scale='RdYlBu_r'
            )
            fig_heat.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0f172a', family='Inter'),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    
    # Gender comparison
    st.markdown("### 👥 Performance by Gender")
    gender_stats = filtered_df.groupby('gender')['total_mark'].agg(['mean', 'count']).round(2)
    gender_stats.columns = ['Average Score', 'Count']
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(gender_stats, use_container_width=True)
    with col2:
        fig_gender = px.bar(
            filtered_df.groupby('gender')['total_mark'].mean().reset_index(),
            x='gender', y='total_mark', color='gender',
            title="Average Score by Gender"
        )
        fig_gender.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gender, use_container_width=True)


# ============== PAGE: STATISTICAL ANALYSIS ==============
elif page == "Statistical Analysis":
    st.title("🧬 Statistical Analysis")
    
    st.markdown("""<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 24px; border-radius: 12px; margin-bottom: 30px;">
        <h3 style="color: white; margin: 0; font-weight: 800;">📊 Hypothesis Testing & Deep Analytics</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0;">Advanced statistical methods for data-driven decisions</p>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("### 📊 Descriptive Statistics")
    numeric_cols = ['exam_1', 'exam_2', 'exam_3', 'total_attendance', 'total_mark']
    available_numeric = [col for col in numeric_cols if col in filtered_df.columns]
    
    if available_numeric:
        desc_stats = filtered_df[available_numeric].describe().round(2)
        st.dataframe(desc_stats, use_container_width=True)
    
    st.markdown("### 🔗 Correlation Matrix")
    
    if len(available_numeric) >= 2:
        corr_matrix = filtered_df[available_numeric].corr()
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto='.2f', 
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap"
        )
        fig_corr.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Hypothesis Testing: Gender
    st.markdown("### 🔬 Hypothesis Testing: Gender vs Performance")
    
    male_scores = filtered_df[filtered_df['gender'] == 'Male']['total_mark']
    female_scores = filtered_df[filtered_df['gender'] == 'Female']['total_mark']
    
    if len(male_scores) > 1 and len(female_scores) > 1:
        t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("T-Statistic", f"{t_stat:.4f}")
        with col2:
            significance = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
            st.metric("P-Value", f"{p_value:.4f}", delta=significance)
        
        if p_value < 0.05:
            st.info("📝 There IS a statistically significant difference between male and female performance.")
        else:
            st.info("📝 There is NO statistically significant difference between male and female performance.")
        
        # Box plot comparison
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=male_scores, name='Male', marker_color='#06b6d4', boxmean='sd'))
        fig_box.add_trace(go.Box(y=female_scores, name='Female', marker_color='#ec4899', boxmean='sd'))
        fig_box.update_layout(
            title="Score Distribution by Gender",
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='#f1f5f9', title='Total Mark'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Normality Test
    st.markdown("### 📈 Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=filtered_df['total_mark'], nbinsx=25, marker_color='#06b6d4'))
        fig_hist.update_layout(
            title="Score Distribution Histogram",
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Score'),
            yaxis=dict(title='Frequency', gridcolor='#f1f5f9'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        if len(filtered_df) >= 20:
            sample_size = min(5000, len(filtered_df))
            stat, p_value = stats.shapiro(filtered_df['total_mark'].sample(sample_size, random_state=42))
            st.metric("Shapiro-Wilk p-value", f"{p_value:.4f}")
            if p_value > 0.05:
                st.success("✅ Data appears normally distributed (p > 0.05)")
            else:
                st.info("ℹ️ Data may not be normally distributed (p ≤ 0.05)")


# ============== PAGE: PREDICTIVE ANALYTICS ==============
elif page == "Predictive Analytics":
    st.title("🔮 Predictive Analytics")
    
    st.markdown("""<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 24px; border-radius: 12px; margin-bottom: 30px;">
        <h3 style="color: white; margin: 0; font-weight: 800;">🚀 Performance Prediction & Segmentation</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0;">ML-powered insights for proactive interventions</p>
    </div>""", unsafe_allow_html=True)
    
    # Trend Analysis
    st.markdown("### 📈 Trend Analysis")
    
    if 'exam_date' in filtered_df.columns and filtered_df['exam_date'].notna().any():
        monthly = filtered_df.groupby(filtered_df['exam_date'].dt.to_period('M'))['total_mark'].mean()
        monthly_df = monthly.reset_index()
        monthly_df['exam_date'] = monthly_df['exam_date'].astype(str)
        
        fig_trend = px.line(monthly_df, x='exam_date', y='total_mark', title="📈 Monthly Average Score Trend", markers=True)
        fig_trend.update_traces(line_color='#06b6d4', line_width=3)
        fig_trend.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter'),
            xaxis=dict(title='Month'),
            yaxis=dict(title='Avg Score', gridcolor='#f1f5f9'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("📅 Date-based trend analysis not available")
    
    # Clustering
    st.markdown("### 🎯 Student Clustering (K-Means)")
    
    if len(filtered_df) >= 10:
        features = ['total_mark', 'total_attendance']
        X = filtered_df[features].dropna()
        
        if len(X) >= 10:
            n_clusters = st.slider("Number of Clusters", 2, 5, 3)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            cluster_df = X.copy()
            cluster_df['Cluster'] = clusters
            
            fig_cluster = px.scatter(
                cluster_df,
                x='total_attendance',
                y='total_mark',
                color='Cluster',
                title="🎯 Student Clusters",
                color_continuous_scale='Viridis'
            )
            fig_cluster.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0f172a', family='Inter'),
                xaxis=dict(title='Attendance', gridcolor='#f1f5f9'),
                yaxis=dict(title='Score', gridcolor='#f1f5f9'),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Cluster statistics
            st.markdown("#### 📊 Cluster Characteristics")
            cluster_stats = cluster_df.groupby('Cluster')[features].mean().round(2)
            cluster_stats['Count'] = cluster_df.groupby('Cluster').size()
            st.dataframe(cluster_stats, use_container_width=True)
            
            st.info("""
            💡 **How to use this:**
            - **High performers** (high score, high attendance) → Advanced programs
            - **At-risk students** (low score, low attendance) → Intensive support
            - **Inconsistent performers** → Motivation programs
            """)
    else:
        st.warning("⚠️ Not enough data for clustering (need at least 10 records)")


# ============== PAGE: RISK ANALYSIS ==============
elif page == "Risk Analysis":
    st.title("🎯 Student Risk Analysis")
    
    st.markdown("""<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 24px; border-radius: 12px; margin-bottom: 30px;">
        <h3 style="color: #7c2d12; margin: 0; font-weight: 800;">⚠️ Proactive Student Support System</h3>
        <p style="color: #78350f; margin: 8px 0 0 0;">Identify at-risk students early</p>
    </div>""", unsafe_allow_html=True)
    
    # Calculate Risk Score
    risk_df = filtered_df.copy()
    risk_df['attendance_risk'] = (10 - risk_df['total_attendance']) * 10
    risk_df['score_risk'] = 100 - risk_df['total_mark']
    risk_df['risk_score'] = (risk_df['attendance_risk'] * 0.4 + risk_df['score_risk'] * 0.6).clip(0, 100)
    
    def categorize_risk(score):
        if score >= 70:
            return "🔴 Critical"
        elif score >= 50:
            return "🟠 High"
        elif score >= 30:
            return "🟡 Medium"
        else:
            return "🟢 Low"
    
    risk_df['risk_category'] = risk_df['risk_score'].apply(categorize_risk)
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    critical = len(risk_df[risk_df['risk_category'] == "🔴 Critical"])
    high = len(risk_df[risk_df['risk_category'] == "🟠 High"])
    medium = len(risk_df[risk_df['risk_category'] == "🟡 Medium"])
    low = len(risk_df[risk_df['risk_category'] == "🟢 Low"])
    
    with col1:
        st.markdown(f"""
        <div style="background: #fee2e2; padding: 20px; border-radius: 10px; border: 2px solid #ef4444;">
            <div style="color: #991b1b; font-size: 0.75rem; font-weight: 700;">CRITICAL RISK</div>
            <div style="color: #dc2626; font-size: 2.5rem; font-weight: 900;">{critical}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #fed7aa; padding: 20px; border-radius: 10px; border: 2px solid #f97316;">
            <div style="color: #7c2d12; font-size: 0.75rem; font-weight: 700;">HIGH RISK</div>
            <div style="color: #ea580c; font-size: 2.5rem; font-weight: 900;">{high}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: #fef3c7; padding: 20px; border-radius: 10px; border: 2px solid #eab308;">
            <div style="color: #713f12; font-size: 0.75rem; font-weight: 700;">MEDIUM RISK</div>
            <div style="color: #ca8a04; font-size: 2.5rem; font-weight: 900;">{medium}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: #d1fae5; padding: 20px; border-radius: 10px; border: 2px solid #22c55e;">
            <div style="color: #14532d; font-size: 0.75rem; font-weight: 700;">LOW RISK</div>
            <div style="color: #16a34a; font-size: 2.5rem; font-weight: 900;">{low}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk distribution chart
    st.markdown("### 📊 Risk Distribution")
    risk_counts = risk_df['risk_category'].value_counts().reset_index()
    risk_counts.columns = ['Category', 'Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_risk = px.pie(risk_counts, values='Count', names='Category', title="Risk Category Distribution", hole=0.4)
        fig_risk.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        fig_scatter_risk = px.scatter(
            risk_df,
            x='total_attendance',
            y='total_mark',
            color='risk_category',
            title="Risk by Score & Attendance",
            opacity=0.7
        )
        fig_scatter_risk.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Attendance', gridcolor='#f1f5f9'),
            yaxis=dict(title='Score', gridcolor='#f1f5f9'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_scatter_risk, use_container_width=True)
    
    # High-risk students table
    st.markdown("### 📋 High-Risk Students (Top 20)")
    at_risk = risk_df[risk_df['risk_score'] >= 50][
        ['student_name', 'subject_name', 'total_mark', 'total_attendance', 'risk_score', 'risk_category']
    ].sort_values('risk_score', ascending=False).head(20)
    
    if not at_risk.empty:
        st.dataframe(at_risk, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No high-risk students found!")


# ============== PAGE: PERFORMANCE BENCHMARKING ==============
elif page == "Performance Benchmarking":
    st.title("🏆 Performance Benchmarking")
    
    st.markdown("""<div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 24px; border-radius: 12px; margin-bottom: 30px;">
        <h3 style="color: #0f172a; margin: 0; font-weight: 800;">📊 Multi-Dimensional Performance Analysis</h3>
        <p style="color: #475569; margin: 8px 0 0 0;">Compare performance across subjects, classes, and time periods</p>
    </div>""", unsafe_allow_html=True)
    
    # Subject Difficulty Analysis
    st.markdown("### 📚 Subject Difficulty Analysis")
    
    subject_perf = filtered_df.groupby('subject_name').agg({
        'total_mark': 'mean',
        'student_name': 'count'
    }).round(2)
    subject_perf.columns = ['Avg Score', 'Students']
    subject_perf['Difficulty'] = (100 - subject_perf['Avg Score']).clip(0, 100)
    subject_perf = subject_perf.sort_values('Difficulty', ascending=False).reset_index()
    
    fig_diff = px.bar(
        subject_perf,
        x='subject_name',
        y='Difficulty',
        color='Difficulty',
        color_continuous_scale='RdYlGn_r',
        title="📚 Subject Difficulty Index (Higher = More Difficult)"
    )
    fig_diff.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0f172a', family='Inter'),
        xaxis=dict(title='Subject'),
        yaxis=dict(title='Difficulty Score', gridcolor='#f1f5f9'),
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_diff, use_container_width=True)
    
    # Top Performers
    st.markdown("### 🏅 Top Performers")
    
    top_students = filtered_df.groupby('student_name')['total_mark'].mean().sort_values(ascending=False).head(10).reset_index()
    top_students.columns = ['Student', 'Avg Score']
    top_students['Rank'] = range(1, len(top_students) + 1)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_top = px.bar(top_students, x='Student', y='Avg Score', title="🏆 Top 10 Students by Average Score")
        fig_top.update_traces(marker_color='#22c55e')
        fig_top.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter'),
            xaxis=dict(title='Student'),
            yaxis=dict(title='Avg Score', gridcolor='#f1f5f9'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        st.dataframe(top_students[['Rank', 'Student', 'Avg Score']], use_container_width=True, hide_index=True)
    
    # Class Comparison
    st.markdown("### 🎓 Class Performance Comparison")
    
    class_perf = filtered_df.groupby('class').agg({
        'total_mark': ['mean', 'std'],
        'total_attendance': 'mean',
        'student_name': 'nunique'
    }).round(2)
    class_perf.columns = ['Avg Score', 'Std Dev', 'Avg Attendance', 'Students']
    class_perf = class_perf.reset_index()
    
    st.dataframe(class_perf, use_container_width=True, hide_index=True)
    
    fig_class = px.bar(class_perf, x='class', y='Avg Score', color='Avg Attendance',
                       title="Class Performance Overview", color_continuous_scale='Blues')
    fig_class.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_class, use_container_width=True)


# ============== PAGE: ML INSIGHTS ==============
elif page == "ML Insights":
    st.title("🤖 Machine Learning Insights")
    
    st.markdown("""
    <div style="background: white; padding: 28px; border-radius: 12px; border: 2px solid #e2e8f0; margin-bottom: 24px;">
        <h3 style="color: #0f172a; margin-top: 0; font-weight: 800;">🎯 Predict Student Performance</h3>
        <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 0;">
            Enter student details to predict their final score using trained ML models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'reg' not in models or 'clf' not in models:
        st.warning("⚠️ ML models are not available. Prediction features are disabled.")
        st.info("""
        **To enable predictions:**
        1. Run the training script locally: `python models/train_model.py`
        2. Push the model files (`*.pkl`) to your repository
        3. Redeploy the application
        """)
        
        # Demo mode
        st.markdown("### 📊 Demo Analysis")
        st.markdown("""
        When models are available, you can:
        - 🎯 **Predict scores** based on attendance and exam performance
        - 📈 **Classify students** into performance categories (High/Medium/Low)
        - 🔮 **Get early warnings** for at-risk students
        """)
        
        # Show basic statistics instead
        st.markdown("### 📈 Current Data Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Score", f"{filtered_df['total_mark'].mean():.1f}")
        with col2:
            st.metric("Average Attendance", f"{filtered_df['total_attendance'].mean():.1f}/10")
        with col3:
            pass_rate = (filtered_df['total_mark'] >= 50).mean() * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
    else:
        st.markdown("### 🎯 Score Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 Input Features")
            attendance = st.slider("Attendance (0-10)", 0, 10, 8)
            exam1 = st.number_input("Exam 1 Score", 0.0, 100.0, 75.0, step=1.0)
            exam2 = st.number_input("Exam 2 Score", 0.0, 100.0, 75.0, step=1.0)
            gender = st.selectbox("Gender", ["Male", "Female"])
            
        with col2:
            st.markdown("#### 🎯 Prediction Results")
            if st.button("🚀 Predict Performance"):
                try:
                    # Regression prediction
                    input_reg = pd.DataFrame([[attendance, exam1, exam2]], 
                                            columns=['total_attendance', 'exam_1', 'exam_2'])
                    pred_score = models['reg'].predict(input_reg)[0]
                    
                    # Classification prediction
                    gender_code = 0 if gender == 'Male' else 1
                    input_clf = pd.DataFrame([[attendance, gender_code, exam1]], 
                                            columns=['total_attendance', 'gender_code', 'exam_1'])
                    pred_group = models['clf'].predict(input_clf)[0]
                    
                    st.success(f"**Predicted Score:** {pred_score:.2f}")
                    
                    category_colors = {
                        'High': '#22c55e',
                        'Medium': '#f59e0b',
                        'Low': '#ef4444'
                    }
                    color = category_colors.get(pred_group, '#6b7280')
                    st.markdown(f"**Performance Category:** <span style='color:{color}; font-weight:bold;'>{pred_group}</span>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")


# ============== PAGE: RAW DATA ==============
elif page == "Raw Data":
    st.title("📄 Raw Data Explorer")
    
    st.markdown(f"### 📊 Dataset Overview ({len(filtered_df):,} records)")
    
    # Search
    search = st.text_input("🔍 Search in data", "")
    
    if search:
        mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
        display_df = filtered_df[mask]
        st.info(f"Found {len(display_df):,} matching records")
    else:
        display_df = filtered_df
    
    # Display with pagination hint
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
    
    # Column info
    with st.expander("📋 Column Information"):
        col_info = pd.DataFrame({
            'Column': display_df.columns,
            'Type': display_df.dtypes.astype(str),
            'Non-Null': display_df.count().values,
            'Unique': display_df.nunique().values
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="student_performance_data.csv",
        mime="text/csv"
    )


# ============== FOOTER ==============
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #94a3b8;">
    <p style="margin: 0;">Built with ❤️ using <strong style="color: #06b6d4;">Streamlit</strong></p>
    <p style="margin: 5px 0 0 0; font-size: 0.85rem;">🎓 Student Performance Analytics Dashboard © 2025</p>
</div>
""", unsafe_allow_html=True)
