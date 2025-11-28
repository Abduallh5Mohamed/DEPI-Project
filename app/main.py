import streamlit as st
import pandas as pd
import sqlite3
import pickle
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from db_utils import create_db_from_csv
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

# Database Connection
DB_PATH = 'student_performance.db'

@st.cache_data
def load_data():
    import traceback
    from pathlib import Path
    # Try to load from SQLite database first. If DB is missing on the deployment
    # platform (e.g. Streamlit Cloud) fallback to the CSV in `data/raw_student_data.csv`.
    try:
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            query = """
            SELECT 
                sc.score_id,
                st.first_name || ' ' || st.last_name as student_name,
                st.gender,
                st.class,
                s.subject_name,
                t.first_name || ' ' || t.last_name as teacher_name,
                sc.exam_1,
                sc.exam_2,
                sc.exam_3,
                sc.total_attendance,
                sc.total_mark,
                sc.exam_date
            FROM scores sc
            JOIN students st ON sc.student_id = st.student_id
            JOIN subjects s ON sc.subject_id = s.subject_id
            JOIN teachers t ON sc.teacher_id = t.teacher_id
            """
            try:
                df = pd.read_sql_query(query, conn)
            except Exception:
                # write detailed traceback to logs/db_errors.log for inspection
                log_dir = Path('logs')
                log_dir.mkdir(exist_ok=True)
                log_file = log_dir / 'db_errors.log'
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write('\n--- DB LOAD ERROR ---\n')
                    f.write(traceback.format_exc())
                conn.close()
                st.warning('Database read failed — falling back to CSV. Details written to logs/db_errors.log')
                df = None
            else:
                conn.close()
                if df is not None and not df.empty:
                    return df
        # CSV fallback
        csv_path = os.path.join('data', 'raw_student_data.csv')
        if os.path.exists(csv_path):
            st.info('Loading data from CSV fallback (data/raw_student_data.csv).')
            df = pd.read_csv(csv_path)
            return df
        else:
            st.error('No data source available: neither database nor CSV found.')
            return pd.DataFrame()
    except Exception:
        # Last-resort catch-all: write traceback and surface friendly message
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'db_errors.log'
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('\n--- UNEXPECTED LOAD ERROR ---\n')
            f.write(traceback.format_exc())
        st.error('Unexpected error while loading data. Details written to logs/db_errors.log')
        return pd.DataFrame()

@st.cache_resource
def load_models():
    models = {}
    try:
        with open('models/score_predictor.pkl', 'rb') as f:
            models['reg'] = pickle.load(f)
        with open('models/performance_classifier.pkl', 'rb') as f:
            models['clf'] = pickle.load(f)
    except FileNotFoundError:
        st.error("Models not found. Please run training script first.")
    return models

# Allow creating a local SQLite DB from CSV via the sidebar before loading data
if st.sidebar.button("🛠️ Initialize DB from CSV (create local SQLite)"):
    success, msg = create_db_from_csv()
    if success:
        st.success(msg + " — restarting app to load database...")
        st.experimental_rerun()
    else:
        st.error("Failed to create DB: " + str(msg))

df = load_data()
models = load_models()

# Sidebar
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
selected_subject = st.sidebar.multiselect("📚 Select Subject", df['subject_name'].unique(), default=df['subject_name'].unique())
selected_class = st.sidebar.multiselect("🎓 Select Class", df['class'].unique(), default=df['class'].unique())

# Filter Data
filtered_df = df[df['subject_name'].isin(selected_subject) & df['class'].isin(selected_class)]

if page == "Overview":
    st.title("🎓 Student Performance Dashboard")
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    avg_score = filtered_df['total_mark'].mean()
    avg_attendance = filtered_df['total_attendance'].mean()
    total_students = filtered_df['student_name'].nunique()
    top_performer = filtered_df.loc[filtered_df['total_mark'].idxmax()]['student_name'] if not filtered_df.empty else "N/A"
    
    with col1:
        st.markdown(card_component("Average Score", f"{avg_score:.1f}", "Across all selected subjects"), unsafe_allow_html=True)
    with col2:
        st.markdown(card_component("Avg Attendance", f"{avg_attendance:.1f}/10", "Lectures attended"), unsafe_allow_html=True)
    with col3:
        st.markdown(card_component("Total Students", f"{total_students}", "Active students"), unsafe_allow_html=True)
    with col4:
        st.markdown(card_component("Top Performer", top_performer.split()[0], "Highest total mark"), unsafe_allow_html=True)

    st.markdown("### 📊 Performance Trends & Analytics")
    
    # Chart: Average Score by Subject
    fig_bar = px.bar(
        filtered_df.groupby('subject_name')['total_mark'].mean().reset_index(),
        x='subject_name',
        y='total_mark',
        title="📊 Average Score by Subject"
    )
    fig_bar.update_traces(
        marker_color='#06b6d4',
        marker_line_color='white',
        marker_line_width=2
    )
    fig_bar.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0f172a', family='Inter', size=13),
        title_font_size=20,
        title_font_color='#0f172a',
        title_font_weight=800,
        xaxis=dict(
            showgrid=False,
            title_font=dict(size=13, color='#64748b', weight=600),
            tickfont=dict(size=12, color='#475569')
        ),
        yaxis=dict(
            gridcolor='#f1f5f9',
            title_font=dict(size=13, color='#64748b', weight=600),
            tickfont=dict(size=12, color='#475569')
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

elif page == "Advanced Insights":
    st.title("📈 Advanced Analytics & Insights")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Subject Performance Gap (Heatmap Logic)
    st.markdown("### 🌡️ Subject Performance vs Global Average")
    st.info("📊 This heatmap shows how each subject performed relative to the school's average for that month. Red indicates above average, blue indicates below average.")
    
    query_gap = """
    SELECT 
        strftime('%Y-%m', sc.exam_date) as month,
        s.subject_name,
        AVG(sc.total_mark) - (SELECT AVG(total_mark) FROM scores s2 
                              WHERE strftime('%Y-%m', s2.exam_date) = strftime('%Y-%m', sc.exam_date)) as performance_gap
    FROM scores sc
    JOIN subjects s ON sc.subject_id = s.subject_id
    GROUP BY month, s.subject_name
    ORDER BY month;
    """
    df_gap = pd.read_sql_query(query_gap, conn)
    
    if not df_gap.empty:
        pivot_gap = df_gap.pivot(index='subject_name', columns='month', values='performance_gap')
        fig_heat = px.imshow(pivot_gap, 
                             labels=dict(x="Month", y="Subject", color="Performance Gap"),
                             x=pivot_gap.columns,
                             y=pivot_gap.index,
                             color_continuous_scale='RdYlBu_r',
                             origin='lower')
        fig_heat.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=20,
            title_font_color='#0f172a',
            title_font_weight=800,
            xaxis=dict(tickfont=dict(size=11, color='#475569')),
            yaxis=dict(tickfont=dict(size=11, color='#475569')),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    col1, col2 = st.columns(2)
    
    with col1:
        # 2. Attendance Volatility
        st.markdown("### 📉 Attendance Volatility")
        st.caption("Variance (Max - Min) in attendance per month.")
        query_vol = """
        SELECT 
            strftime('%Y-%m', exam_date) as month,
            (MAX(total_attendance) - MIN(total_attendance)) as attendance_variance
        FROM scores
        GROUP BY month
        ORDER BY month;
        """
        df_vol = pd.read_sql_query(query_vol, conn)
        fig_vol = px.bar(df_vol, x='month', y='attendance_variance', title="📅 Monthly Attendance Variance")
        fig_vol.update_traces(marker_color='#14b8a6', marker_line_color='white', marker_line_width=2)
        fig_vol.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=18,
            title_font_color='#0f172a',
            title_font_weight=800,
            xaxis=dict(
                showgrid=False,
                title_font=dict(size=12, color='#64748b'),
                tickfont=dict(size=11, color='#475569')
            ),
            yaxis=dict(
                gridcolor='#f1f5f9',
                title_font=dict(size=12, color='#64748b'),
                tickfont=dict(size=11, color='#475569')
            ),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with col2:
        # 3. Teacher Impact (Expanded)
        st.markdown("### 👨‍🏫 Teacher Impact Analysis")
        
        tab1, tab2, tab3 = st.tabs(["At-Risk Support", "Value Added", "Consistency"])
        
        with tab1:
            st.caption("Avg score of students with < 5 attendance.")
            query_risk = """
            SELECT 
                t.first_name || ' ' || t.last_name as teacher_name,
                AVG(sc.total_mark) as avg_score_for_at_risk
            FROM scores sc
            JOIN teachers t ON sc.teacher_id = t.teacher_id
            WHERE sc.student_id IN (
                SELECT student_id FROM scores WHERE total_attendance < 5
            )
            GROUP BY t.teacher_id
            HAVING COUNT(sc.score_id) > 5
            ORDER BY avg_score_for_at_risk DESC
            LIMIT 5;
            """
            df_risk = pd.read_sql_query(query_risk, conn)
            fig_risk = px.bar(df_risk, x='avg_score_for_at_risk', y='teacher_name', orientation='h', 
                             title="👨‍🏫 Best Teachers for At-Risk Students")
            fig_risk.update_traces(marker_color='#0891b2', marker_line_color='white', marker_line_width=1.5)
            fig_risk.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0f172a', family='Inter', size=13),
                title_font_size=16,
                title_font_color='#0f172a',
                title_font_weight=800,
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#f1f5f9',
                    title_font=dict(size=12, color='#64748b'),
                    tickfont=dict(size=11, color='#475569')
                ),
                yaxis=dict(
                    showgrid=False,
                    title_font=dict(size=12, color='#64748b'),
                    tickfont=dict(size=11, color='#475569')
                ),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
        with tab2:
            st.caption("Performance relative to subject average.")
            query_value = """
            SELECT 
                t.first_name || ' ' || t.last_name as teacher_name,
                s.subject_name,
                AVG(sc.total_mark) - (SELECT AVG(total_mark) FROM scores s2 WHERE s2.subject_id = sc.subject_id) as value_added
            FROM scores sc
            JOIN teachers t ON sc.teacher_id = t.teacher_id
            JOIN subjects s ON sc.subject_id = s.subject_id
            GROUP BY t.teacher_id, s.subject_name
            ORDER BY value_added DESC
            LIMIT 10;
            """
            df_value = pd.read_sql_query(query_value, conn)
            fig_val = px.scatter(df_value, x='value_added', y='teacher_name', color='subject_name', 
                                size_max=15, title="⭐ Top Teachers by Value Added",
                                color_discrete_sequence=px.colors.qualitative.Set2)
            fig_val.add_vline(x=0, line_dash="dash", line_color="#cbd5e1", line_width=2)
            fig_val.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0f172a', family='Inter', size=13),
                title_font_size=16,
                title_font_color='#0f172a',
                title_font_weight=800,
                xaxis=dict(
                    gridcolor='#f1f5f9',
                    title_font=dict(size=12, color='#64748b'),
                    tickfont=dict(size=11, color='#475569')
                ),
                yaxis=dict(
                    showgrid=False,
                    title_font=dict(size=12, color='#64748b'),
                    tickfont=dict(size=11, color='#475569')
                ),
                legend=dict(
                    bgcolor='white',
                    bordercolor='#e2e8f0',
                    borderwidth=2,
                    font=dict(size=11, color='#475569')
                ),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_val, use_container_width=True)
            
        with tab3:
            st.caption("Consistency (Lower deviation = More consistent).")
            query_cons = """
            SELECT 
                t.first_name || ' ' || t.last_name as teacher_name,
                AVG(ABS(sc.total_mark - (SELECT AVG(total_mark) FROM scores s2 WHERE s2.teacher_id = t.teacher_id))) as avg_deviation
            FROM scores sc
            JOIN teachers t ON sc.teacher_id = t.teacher_id
            GROUP BY t.teacher_id
            HAVING COUNT(sc.score_id) > 10
            ORDER BY avg_deviation ASC
            LIMIT 5;
            """
            df_cons = pd.read_sql_query(query_cons, conn)
            st.table(df_cons)
    
    conn.close()

elif page == "Statistical Analysis":
    st.title("🧬 Advanced Statistical Analysis")
    
    st.markdown("""<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 24px; border-radius: 12px; margin-bottom: 30px;">
        <h3 style="color: white; margin: 0; font-weight: 800;">📊 Statistical Hypothesis Testing & Deep Analytics</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 0.95rem;">Advanced statistical methods for data-driven decision making</p>
    </div>""", unsafe_allow_html=True)
    
    # 1. Hypothesis Testing: Gender Performance
    st.markdown("### 🔬 Hypothesis Testing: Gender vs Performance")
    col1, col2 = st.columns(2)
    
    male_scores = filtered_df[filtered_df['gender'] == 'Male']['total_mark']
    female_scores = filtered_df[filtered_df['gender'] == 'Female']['total_mark']
    
    if len(male_scores) > 0 and len(female_scores) > 0:
        t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
        
        with col1:
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; border: 2px solid #06b6d4;">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">T-Statistic</div>
                <div style="color: #0f172a; font-size: 2.5rem; font-weight: 900; margin: 8px 0;">{t_stat:.4f}</div>
                <div style="color: #64748b; font-size: 0.85rem;">Difference magnitude</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            significance = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
            color = "#22c55e" if p_value < 0.05 else "#ef4444"
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; border: 2px solid {color};">
                <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">P-Value</div>
                <div style="color: {color}; font-size: 2.5rem; font-weight: 900; margin: 8px 0;">{p_value:.4f}</div>
                <div style="color: #64748b; font-size: 0.85rem;">{significance} (α=0.05)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info(f"📝 **Interpretation:** {'There IS a statistically significant difference between male and female performance.' if p_value < 0.05 else 'There is NO statistically significant difference between male and female performance.'}")
        
        # Box plot comparison
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=male_scores, name='Male', marker_color='#06b6d4', boxmean='sd'))
        fig_box.add_trace(go.Box(y=female_scores, name='Female', marker_color='#ec4899', boxmean='sd'))
        fig_box.update_layout(
            title="📊 Score Distribution by Gender (with Mean & SD)",
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=18,
            title_font_color='#0f172a',
            title_font_weight=800,
            yaxis=dict(gridcolor='#f1f5f9', title='Total Mark'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Correlation Matrix
    st.markdown("### 🔗 Correlation Matrix Analysis")
    st.caption("Understanding relationships between numerical variables")
    
    numeric_cols = ['exam_1', 'exam_2', 'exam_3', 'total_attendance', 'total_mark']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                         text_auto='.2f',
                         color_continuous_scale='RdBu_r',
                         aspect='auto',
                         labels=dict(color="Correlation"))
    fig_corr.update_layout(
        title="🔗 Correlation Heatmap",
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0f172a', family='Inter', size=12),
        title_font_size=18,
        title_font_color='#0f172a',
        title_font_weight=800,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Find strongest correlations
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Variable 1': corr_matrix.columns[i],
                'Variable 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔥 Top 3 Positive Correlations**")
        st.dataframe(corr_df.head(3), hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**❄️ Top 3 Negative Correlations**")
        st.dataframe(corr_df.tail(3), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # 3. Anomaly Detection (Z-Score Method)
    st.markdown("### 🚨 Anomaly Detection")
    st.caption("Identify students with unusual performance patterns (|Z-score| > 2.5)")
    
    filtered_df_copy = filtered_df.copy()
    filtered_df_copy['z_score'] = np.abs(stats.zscore(filtered_df_copy['total_mark']))
    anomalies = filtered_df_copy[filtered_df_copy['z_score'] > 2.5].sort_values('z_score', ascending=False)
    
    if len(anomalies) > 0:
        st.warning(f"⚠️ Found {len(anomalies)} anomalous records")
        st.dataframe(anomalies[['student_name', 'subject_name', 'total_mark', 'z_score']], use_container_width=True)
        
        # Visualization
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(
            x=filtered_df_copy.index,
            y=filtered_df_copy['total_mark'],
            mode='markers',
            name='Normal',
            marker=dict(color='#06b6d4', size=6, opacity=0.6)
        ))
        fig_anom.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies['total_mark'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#ef4444', size=12, symbol='star', line=dict(color='#dc2626', width=2))
        ))
        fig_anom.update_layout(
            title="🎯 Score Distribution with Anomalies Highlighted",
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=18,
            title_font_color='#0f172a',
            title_font_weight=800,
            yaxis=dict(gridcolor='#f1f5f9', title='Total Mark'),
            xaxis=dict(title='Record Index'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_anom, use_container_width=True)
    else:
        st.success("✅ No significant anomalies detected in the data")
    
    st.markdown("---")
    
    # 4. Distribution Analysis
    st.markdown("### 📊 Distribution Analysis & Normality Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram with KDE
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=filtered_df['total_mark'],
            nbinsx=30,
            name='Frequency',
            marker_color='#06b6d4',
            opacity=0.7
        ))
        fig_hist.update_layout(
            title="📈 Score Distribution Histogram",
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=16,
            title_font_color='#0f172a',
            title_font_weight=800,
            yaxis=dict(gridcolor='#f1f5f9', title='Frequency'),
            xaxis=dict(title='Total Mark'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Q-Q Plot
        from scipy.stats import probplot
        qq_data = probplot(filtered_df['total_mark'], dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[0][1],
            mode='markers',
            name='Data',
            marker=dict(color='#06b6d4', size=6)
        ))
        fig_qq.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1],
            mode='lines',
            name='Normal',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
        fig_qq.update_layout(
            title="📐 Q-Q Plot (Normality Check)",
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=16,
            title_font_color='#0f172a',
            title_font_weight=800,
            yaxis=dict(gridcolor='#f1f5f9', title='Sample Quantiles'),
            xaxis=dict(gridcolor='#f1f5f9', title='Theoretical Quantiles'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_qq, use_container_width=True)
    
    # Shapiro-Wilk Test
    if len(filtered_df) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(filtered_df['total_mark'].sample(min(5000, len(filtered_df))))
        normal = "✅ Normally Distributed" if shapiro_p > 0.05 else "❌ Not Normally Distributed"
        st.info(f"**Shapiro-Wilk Test:** p-value = {shapiro_p:.4f} → {normal} (α=0.05)")

elif page == "Predictive Analytics":
    st.title("🔮 Predictive Analytics & Forecasting")
    
    st.markdown("""<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 24px; border-radius: 12px; margin-bottom: 30px;">
        <h3 style="color: white; margin: 0; font-weight: 800;">🚀 Future Performance Prediction & Student Segmentation</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 0.95rem;">Machine learning-powered insights for proactive interventions</p>
    </div>""", unsafe_allow_html=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Time Series Analysis
    st.markdown("### 📈 Performance Trends Over Time")
    
    query_ts = """
    SELECT 
        strftime('%Y-%m', exam_date) as month,
        AVG(total_mark) as avg_score,
        AVG(total_attendance) as avg_attendance,
        COUNT(*) as num_students
    FROM scores
    GROUP BY month
    ORDER BY month;
    """
    df_ts = pd.read_sql_query(query_ts, conn)
    
    if len(df_ts) > 0:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_ts['month'],
            y=df_ts['avg_score'],
            mode='lines+markers',
            name='Avg Score',
            line=dict(color='#06b6d4', width=3),
            marker=dict(size=10, symbol='circle', line=dict(color='white', width=2))
        ))
        
        # Add trend line
        from numpy.polynomial import polynomial as P
        x_numeric = np.arange(len(df_ts))
        coefs = P.polyfit(x_numeric, df_ts['avg_score'], 1)
        trend_line = P.polyval(x_numeric, coefs)
        
        fig_ts.add_trace(go.Scatter(
            x=df_ts['month'],
            y=trend_line,
            mode='lines',
            name='Trend',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
        
        fig_ts.update_layout(
            title="📊 Average Score Trend with Linear Regression",
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=18,
            title_font_color='#0f172a',
            title_font_weight=800,
            yaxis=dict(gridcolor='#f1f5f9', title='Average Score'),
            xaxis=dict(title='Month'),
            legend=dict(bgcolor='white', bordercolor='#e2e8f0', borderwidth=2),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Forecast next 3 months
        st.markdown("#### 🔮 Simple Forecast (Next 3 Months)")
        next_months = ['Next-1', 'Next-2', 'Next-3']
        x_future = np.arange(len(df_ts), len(df_ts) + 3)
        forecast = P.polyval(x_future, coefs)
        
        forecast_df = pd.DataFrame({
            'Period': next_months,
            'Predicted Avg Score': forecast
        })
        
        col1, col2, col3 = st.columns(3)
        for idx, (col, row) in enumerate(zip([col1, col2, col3], forecast_df.itertuples())):
            with col:
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 10px; border: 2px solid #06b6d4;">
                    <div style="color: #64748b; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">{row.Period}</div>
                    <div style="color: #06b6d4; font-size: 2rem; font-weight: 900; margin: 8px 0;">{row._2:.1f}</div>
                    <div style="color: #64748b; font-size: 0.85rem;">Predicted Score</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Student Segmentation with K-Means
    st.markdown("### 🎯 Student Segmentation (K-Means Clustering)")
    st.caption("Automatically group students with similar characteristics for targeted interventions")
    
    query_seg = """
    SELECT 
        student_id,
        AVG(total_mark) as avg_score,
        AVG(total_attendance) as avg_attendance,
        AVG((exam_1 + exam_2 + exam_3)/3) as avg_exam_score
    FROM scores
    GROUP BY student_id
    HAVING COUNT(*) >= 3;
    """
    df_seg = pd.read_sql_query(query_seg, conn)
    
    if len(df_seg) > 10:
        # Prepare data
        features = ['avg_score', 'avg_attendance', 'avg_exam_score']
        X = df_seg[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means clustering
        n_clusters = st.slider("Number of Clusters", 2, 6, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_seg['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_seg['PC1'] = X_pca[:, 0]
        df_seg['PC2'] = X_pca[:, 1]
        
        fig_cluster = px.scatter(
            df_seg,
            x='PC1',
            y='PC2',
            color='Cluster',
            title="🔍 Student Clusters (PCA Visualization)",
            color_continuous_scale='Viridis',
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'},
            hover_data=['avg_score', 'avg_attendance']
        )
        fig_cluster.update_traces(marker=dict(size=10, line=dict(color='white', width=1)))
        fig_cluster.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=18,
            title_font_color='#0f172a',
            title_font_weight=800,
            xaxis=dict(gridcolor='#f1f5f9'),
            yaxis=dict(gridcolor='#f1f5f9'),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Cluster Statistics
        st.markdown("#### 📊 Cluster Characteristics")
        cluster_stats = df_seg.groupby('Cluster')[features].mean().round(2)
        cluster_stats['Count'] = df_seg.groupby('Cluster').size()
        cluster_stats = cluster_stats.reset_index()
        cluster_stats.columns = ['Cluster', 'Avg Score', 'Avg Attendance', 'Avg Exam Score', 'Students']
        st.dataframe(cluster_stats, use_container_width=True, hide_index=True)
        
        st.info("""
        💡 **How to use this:**
        - **High performers** (high score, high attendance) → Advanced programs
        - **At-risk students** (low score, low attendance) → Intensive support
        - **Inconsistent performers** (moderate scores, variable attendance) → Motivation programs
        """)
    else:
        st.warning("⚠️ Not enough data for clustering analysis (need at least 10 students with 3+ records)")
    
    st.markdown("---")
    
    # 3. Cohort Analysis
    st.markdown("### 👥 Cohort Analysis by Class")
    st.caption("Track performance evolution across different student classes")
    
    query_cohort = """
    SELECT 
        st.class,
        strftime('%Y-%m', sc.exam_date) as month,
        AVG(sc.total_mark) as avg_score
    FROM scores sc
    JOIN students st ON sc.student_id = st.student_id
    GROUP BY st.class, month
    ORDER BY st.class, month;
    """
    df_cohort = pd.read_sql_query(query_cohort, conn)
    
    if len(df_cohort) > 0:
        fig_cohort = px.line(
            df_cohort,
            x='month',
            y='avg_score',
            color='class',
            title="📚 Performance Evolution by Class",
            markers=True
        )
        fig_cohort.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(color='white', width=2)))
        fig_cohort.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=18,
            title_font_color='#0f172a',
            title_font_weight=800,
            yaxis=dict(gridcolor='#f1f5f9', title='Average Score'),
            xaxis=dict(title='Month'),
            legend=dict(title='Class', bgcolor='white', bordercolor='#e2e8f0', borderwidth=2),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_cohort, use_container_width=True)
        
        # Class Performance Summary
        class_summary = df_cohort.groupby('class').agg({
            'avg_score': ['mean', 'std', 'min', 'max']
        }).round(2)
        class_summary.columns = ['Mean Score', 'Std Dev', 'Min Score', 'Max Score']
        st.dataframe(class_summary, use_container_width=True)
    
    conn.close()

elif page == "Risk Analysis":
    st.title("🎯 Student Risk Analysis & Early Warning System")
    
    st.markdown("""<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 24px; border-radius: 12px; margin-bottom: 30px;">
        <h3 style="color: #7c2d12; margin: 0; font-weight: 800;">⚠️ Proactive Student Support System</h3>
        <p style="color: #78350f; margin: 8px 0 0 0; font-size: 0.95rem;">Identify at-risk students before it's too late</p>
    </div>""", unsafe_allow_html=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Calculate Risk Score
    query_risk = """
    SELECT 
        st.student_id,
        st.first_name || ' ' || st.last_name as student_name,
        st.class,
        st.gender,
        AVG(sc.total_mark) as avg_score,
        AVG(sc.total_attendance) as avg_attendance,
        COUNT(sc.score_id) as num_exams,
        MIN(sc.total_mark) as lowest_score,
        MAX(sc.total_mark) as highest_score,
        (MAX(sc.total_mark) - MIN(sc.total_mark)) as score_volatility
    FROM students st
    JOIN scores sc ON st.student_id = sc.student_id
    GROUP BY st.student_id;
    """
    df_risk = pd.read_sql_query(query_risk, conn)
    
    # Risk Score Calculation (0-100, higher = more at risk)
    df_risk['attendance_risk'] = (10 - df_risk['avg_attendance']) * 10  # Max 100
    df_risk['score_risk'] = (100 - df_risk['avg_score'])  # Max 100
    df_risk['volatility_risk'] = (df_risk['score_volatility'] / 100) * 100  # Normalized
    df_risk['risk_score'] = (
        df_risk['attendance_risk'] * 0.4 + 
        df_risk['score_risk'] * 0.4 + 
        df_risk['volatility_risk'] * 0.2
    ).clip(0, 100)
    
    # Risk Categories
    def categorize_risk(score):
        if score >= 70:
            return "🔴 Critical"
        elif score >= 50:
            return "🟠 High"
        elif score >= 30:
            return "🟡 Medium"
        else:
            return "🟢 Low"
    
    df_risk['risk_category'] = df_risk['risk_score'].apply(categorize_risk)
    df_risk = df_risk.sort_values('risk_score', ascending=False)
    
    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    critical = len(df_risk[df_risk['risk_category'] == "🔴 Critical"])
    high = len(df_risk[df_risk['risk_category'] == "🟠 High"])
    medium = len(df_risk[df_risk['risk_category'] == "🟡 Medium"])
    low = len(df_risk[df_risk['risk_category'] == "🟢 Low"])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 20px; border-radius: 10px; border: 2px solid #ef4444;">
            <div style="color: #991b1b; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Critical Risk</div>
            <div style="color: #dc2626; font-size: 2.5rem; font-weight: 900; margin: 8px 0;">{critical}</div>
            <div style="color: #991b1b; font-size: 0.85rem;">Immediate action needed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%); padding: 20px; border-radius: 10px; border: 2px solid #f97316;">
            <div style="color: #7c2d12; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">High Risk</div>
            <div style="color: #ea580c; font-size: 2.5rem; font-weight: 900; margin: 8px 0;">{high}</div>
            <div style="color: #7c2d12; font-size: 0.85rem;">Close monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 20px; border-radius: 10px; border: 2px solid #eab308;">
            <div style="color: #713f12; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Medium Risk</div>
            <div style="color: #ca8a04; font-size: 2.5rem; font-weight: 900; margin: 8px 0;">{medium}</div>
            <div style="color: #713f12; font-size: 0.85rem;">Regular check-ins</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 20px; border-radius: 10px; border: 2px solid #22c55e;">
            <div style="color: #14532d; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Low Risk</div>
            <div style="color: #16a34a; font-size: 2.5rem; font-weight: 900; margin: 8px 0;">{low}</div>
            <div style="color: #14532d; font-size: 0.85rem;">On track</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚨 Priority Intervention List")
    st.caption("Top 15 students requiring immediate attention")
    
    priority_students = df_risk.head(15)[[
        'student_name', 'class', 'risk_category', 'risk_score', 
        'avg_score', 'avg_attendance', 'score_volatility'
    ]].round(2)
    
    st.dataframe(
        priority_students,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
        }
    )
    
    # Risk Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Risk Score Distribution")
        fig_hist_risk = px.histogram(
            df_risk, 
            x='risk_score', 
            nbins=20,
            color_discrete_sequence=['#ef4444']
        )
        fig_hist_risk.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            xaxis=dict(gridcolor='#f1f5f9', title='Risk Score'),
            yaxis=dict(gridcolor='#f1f5f9', title='Count'),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_hist_risk, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Risk by Class")
        class_risk = df_risk.groupby('class')['risk_score'].mean().reset_index()
        fig_bar_class = px.bar(
            class_risk,
            x='class',
            y='risk_score',
            color='risk_score',
            color_continuous_scale='Reds'
        )
        fig_bar_class.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            xaxis=dict(showgrid=False, title='Class'),
            yaxis=dict(gridcolor='#f1f5f9', title='Avg Risk Score'),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_bar_class, use_container_width=True)
    
    # Detailed Risk Analysis
    st.markdown("### 🔍 Detailed Risk Factor Analysis")
    selected_student = st.selectbox(
        "Select a student for detailed analysis",
        df_risk['student_name'].tolist()
    )
    
    if selected_student:
        student_data = df_risk[df_risk['student_name'] == selected_student].iloc[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Risk Factors Breakdown
            factors = {
                'Attendance Risk': student_data['attendance_risk'],
                'Score Risk': student_data['score_risk'],
                'Volatility Risk': student_data['volatility_risk']
            }
            
            fig_factors = go.Figure(data=[
                go.Bar(
                    x=list(factors.keys()),
                    y=list(factors.values()),
                    marker_color=['#3b82f6', '#06b6d4', '#8b5cf6'],
                    text=[f"{v:.1f}" for v in factors.values()],
                    textposition='outside'
                )
            ])
            fig_factors.update_layout(
                title=f"Risk Factors for {selected_student}",
                plot_bgcolor='white',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0f172a', family='Inter', size=13),
                title_font_size=16,
                title_font_weight=800,
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='#f1f5f9', title='Risk Score', range=[0, 100]),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_factors, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Student Profile")
            st.markdown(f"""
            **Class:** {student_data['class']}  
            **Gender:** {student_data['gender']}  
            **Total Exams:** {int(student_data['num_exams'])}  
            **Avg Score:** {student_data['avg_score']:.1f}  
            **Avg Attendance:** {student_data['avg_attendance']:.1f}/10  
            **Score Range:** {student_data['lowest_score']:.0f} - {student_data['highest_score']:.0f}
            """)
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            if student_data['risk_score'] >= 70:
                st.error("""
                **Immediate Actions:**
                - Schedule parent meeting
                - Assign peer tutor
                - Daily attendance monitoring
                - Weekly progress check-ins
                """)
            elif student_data['risk_score'] >= 50:
                st.warning("""
                **Suggested Actions:**
                - Additional tutoring sessions
                - Attendance incentives
                - Bi-weekly progress reviews
                """)
            elif student_data['risk_score'] >= 30:
                st.info("""
                **Monitoring Plan:**
                - Monthly check-ins
                - Study skills workshop
                - Encourage peer study groups
                """)
            else:
                st.success("""
                **Maintenance:**
                - Continue current approach
                - Offer advanced challenges
                - Consider mentorship role
                """)
    
    conn.close()

elif page == "Performance Benchmarking":
    st.title("🏆 Performance Benchmarking & Comparison")
    
    st.markdown("""<div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 24px; border-radius: 12px; margin-bottom: 30px;">
        <h3 style="color: #0f172a; margin: 0; font-weight: 800;">📊 Multi-Dimensional Performance Analysis</h3>
        <p style="color: #475569; margin: 8px 0 0 0; font-size: 0.95rem;">Compare performance across subjects, classes, and time periods</p>
    </div>""", unsafe_allow_html=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # What-If Analysis Tool
    st.markdown("### 🧪 What-If Analysis")
    st.caption("Simulate the impact of interventions on student outcomes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_avg = filtered_df['total_mark'].mean()
        st.metric("Current Average Score", f"{current_avg:.1f}")
    
    with col2:
        attendance_boost = st.slider("Attendance Improvement (%)", 0, 50, 10)
        improved_avg = current_avg * (1 + attendance_boost/100 * 0.3)  # Assuming 30% correlation
        st.metric(
            "Projected Average", 
            f"{improved_avg:.1f}",
            delta=f"+{improved_avg-current_avg:.1f}"
        )
    
    with col3:
        students_affected = int(len(filtered_df['student_name'].unique()) * attendance_boost / 100)
        st.metric("Students Impacted", students_affected)
    
    st.info(f"💡 **Insight:** Improving attendance by {attendance_boost}% could raise the average score by {improved_avg-current_avg:.1f} points, affecting {students_affected} students.")
    
    st.markdown("---")
    
    # Subject Difficulty Index
    st.markdown("### 📚 Subject Difficulty Analysis")
    st.caption("Identify challenging subjects based on average scores and pass rates")
    
    query_difficulty = """
    SELECT 
        s.subject_name,
        AVG(sc.total_mark) as avg_score,
        COUNT(CASE WHEN sc.total_mark >= 60 THEN 1 END) * 100.0 / COUNT(*) as pass_rate,
        COUNT(*) as total_students
    FROM scores sc
    JOIN subjects s ON sc.subject_id = s.subject_id
    GROUP BY s.subject_name;
    """
    df_difficulty = pd.read_sql_query(query_difficulty, conn)
    
    # Calculate standard deviation using pandas
    query_std = """
    SELECT 
        s.subject_name,
        sc.total_mark
    FROM scores sc
    JOIN subjects s ON sc.subject_id = s.subject_id;
    """
    df_std = pd.read_sql_query(query_std, conn)
    std_by_subject = df_std.groupby('subject_name')['total_mark'].std().reset_index()
    std_by_subject.columns = ['subject_name', 'score_std']
    df_difficulty = df_difficulty.merge(std_by_subject, on='subject_name', how='left')
    
    # Calculate difficulty score (lower avg + lower pass rate = harder)
    df_difficulty['difficulty_score'] = (
        (100 - df_difficulty['avg_score']) * 0.5 + 
        (100 - df_difficulty['pass_rate']) * 0.5
    )
    df_difficulty = df_difficulty.sort_values('difficulty_score', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_diff = px.bar(
            df_difficulty,
            x='subject_name',
            y='difficulty_score',
            color='difficulty_score',
            color_continuous_scale='RdYlGn_r',
            title="📊 Subject Difficulty Index"
        )
        fig_diff.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=16,
            title_font_weight=800,
            xaxis=dict(showgrid=False, title='Subject'),
            yaxis=dict(gridcolor='#f1f5f9', title='Difficulty Score'),
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_diff, use_container_width=True)
    
    with col2:
        fig_scatter_diff = px.scatter(
            df_difficulty,
            x='avg_score',
            y='pass_rate',
            size='total_students',
            color='subject_name',
            text='subject_name',
            title="🎯 Score vs Pass Rate"
        )
        fig_scatter_diff.update_traces(textposition='top center')
        fig_scatter_diff.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#0f172a', family='Inter', size=13),
            title_font_size=16,
            title_font_weight=800,
            xaxis=dict(gridcolor='#f1f5f9', title='Average Score'),
            yaxis=dict(gridcolor='#f1f5f9', title='Pass Rate (%)'),
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_scatter_diff, use_container_width=True)
    
    st.markdown("---")
    
    # Percentile Rankings
    st.markdown("### 📊 Student Percentile Rankings")
    st.caption("See where students stand relative to their peers")
    
    query_rankings = """
    SELECT 
        st.first_name || ' ' || st.last_name as student_name,
        st.class,
        AVG(sc.total_mark) as avg_score
    FROM students st
    JOIN scores sc ON st.student_id = sc.student_id
    GROUP BY st.student_id
    ORDER BY avg_score DESC;
    """
    df_rankings = pd.read_sql_query(query_rankings, conn)
    df_rankings['percentile'] = df_rankings['avg_score'].rank(pct=True) * 100
    df_rankings['rank'] = range(1, len(df_rankings) + 1)
    
    # Performance Categories
    def get_performance_tier(percentile):
        if percentile >= 90:
            return "⭐ Elite (Top 10%)"
        elif percentile >= 75:
            return "🏅 High Performer (Top 25%)"
        elif percentile >= 50:
            return "✅ Above Average"
        elif percentile >= 25:
            return "📊 Below Average"
        else:
            return "⚠️ Needs Support (Bottom 25%)"
    
    df_rankings['tier'] = df_rankings['percentile'].apply(get_performance_tier)
    
    # Display top and bottom performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Top 10 Performers")
        top_10 = df_rankings.head(10)[['rank', 'student_name', 'class', 'avg_score', 'percentile']].round(2)
        st.dataframe(top_10, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 📉 Bottom 10 Performers")
        bottom_10 = df_rankings.tail(10)[['rank', 'student_name', 'class', 'avg_score', 'percentile']].round(2)
        st.dataframe(bottom_10, use_container_width=True, hide_index=True)
    
    # Tier Distribution
    st.markdown("### 🎯 Performance Tier Distribution")
    tier_counts = df_rankings['tier'].value_counts().reset_index()
    tier_counts.columns = ['Tier', 'Count']
    
    fig_tiers = px.pie(
        tier_counts,
        values='Count',
        names='Tier',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_tiers.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#0f172a', family='Inter', size=13),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_tiers, use_container_width=True)
    
    # Interactive Student Lookup
    st.markdown("### 🔍 Individual Student Lookup")
    search_student = st.selectbox(
        "Search for a student",
        df_rankings['student_name'].tolist()
    )
    
    if search_student:
        student_info = df_rankings[df_rankings['student_name'] == search_student].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rank", f"#{int(student_info['rank'])}")
        with col2:
            st.metric("Percentile", f"{student_info['percentile']:.1f}%")
        with col3:
            st.metric("Avg Score", f"{student_info['avg_score']:.1f}")
        with col4:
            st.metric("Class", student_info['class'])
        
        st.info(f"🎯 **Performance Tier:** {student_info['tier']}")
    
    conn.close()

elif page == "ML Insights":
    st.title("🤖 Machine Learning Insights")
    
    st.markdown("""
    <div style="background: white; padding: 28px; border-radius: 12px; border: 2px solid #e2e8f0; margin-bottom: 24px;">
        <h3 style="color: #0f172a; margin-top: 0; font-weight: 800;">🎯 Predict Student Performance</h3>
        <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 0; line-height: 1.6;">
            Enter student details below to predict their final score and performance category using our trained ML models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
            if 'reg' in models and 'clf' in models:
                # Prepare input
                # Regression input: ['total_attendance', 'exam_1', 'exam_2']
                input_reg = pd.DataFrame([[attendance, exam1, exam2]], columns=['total_attendance', 'exam_1', 'exam_2'])
                pred_score = models['reg'].predict(input_reg)[0]
                
                # Classification input: ['total_attendance', 'gender_code', 'exam_1']
                gender_code = 0 if gender == 'Male' else 1
                input_clf = pd.DataFrame([[attendance, gender_code, exam1]], columns=['total_attendance', 'gender_code', 'exam_1'])
                pred_group = models['clf'].predict(input_clf)[0]
                
                st.markdown(f"""
                <div style="background: #ecfdf5; padding: 24px; border-radius: 10px; border: 2px solid #22c55e; margin-top: 16px;">
                    <div style="color: #166534; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px;">Predicted Total Score</div>
                    <div style="color: #16a34a; font-size: 2.8rem; font-weight: 900; margin: 10px 0; letter-spacing: -1px;">{pred_score:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                category_colors = {
                    'High': ('#ecfdf5', '#22c55e', '#16a34a'),
                    'Medium': ('#fef3c7', '#f59e0b', '#d97706'),
                    'Low': ('#fee2e2', '#ef4444', '#dc2626')
                }
                
                bg, border, text = category_colors.get(pred_group, category_colors['Medium'])
                
                st.markdown(f"""
                <div style="background: {bg}; padding: 24px; border-radius: 10px; border: 2px solid {border}; margin-top: 16px;">
                    <div style="color: {text}; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px;">Performance Category</div>
                    <div style="color: {text}; font-size: 2.8rem; font-weight: 900; margin: 10px 0; letter-spacing: -1px;">{pred_group}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Models not loaded. Please train models first.")

elif page == "Raw Data":
    st.title("📊 Raw Data Explorer")
    st.markdown("### 🔍 Complete Dataset View")
    st.dataframe(filtered_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 24px; color: #94a3b8; font-size: 0.9rem;">
    <p style="margin: 0; font-weight: 600;">Built with ❤️ using <strong style="color: #06b6d4;">Streamlit</strong>, <strong style="color: #06b6d4;">Python</strong>, and <strong style="color: #06b6d4;">Machine Learning</strong></p>
    <p style="margin: 10px 0 0 0; font-size: 0.85rem; color: #cbd5e1;">🎓 Student Performance Analytics Dashboard © 2025</p>
</div>
""", unsafe_allow_html=True)
