def get_custom_css():
    return """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        /* ===================== CLEAN MODERN DESIGN ===================== */
        
        .stApp {
            background: #ffffff;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #0f172a;
        }

        /* ===================== TYPOGRAPHY ===================== */
        
        h1 {
            font-family: 'Inter', sans-serif;
            font-weight: 900 !important;
            font-size: 2.75rem !important;
            color: #0f172a;
            margin-bottom: 0.75rem !important;
            letter-spacing: -1.5px;
        }
        
        h2 {
            font-family: 'Inter', sans-serif;
            font-weight: 800 !important;
            font-size: 1.75rem !important;
            color: #1e293b;
            margin: 2rem 0 1rem 0 !important;
            letter-spacing: -0.5px;
        }
        
        h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 700 !important;
            font-size: 1.25rem !important;
            color: #334155;
            margin: 1.5rem 0 1rem 0 !important;
        }
        
        h4, h5, h6 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #475569;
        }
        
        p, span, div, label {
            color: #64748b;
            line-height: 1.6;
        }

        /* ===================== MODERN MINIMAL CARDS ===================== */
        
        .neon-card {
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 20px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        .neon-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            border-color: #22d3ee;
        }

        /* ===================== METRICS WITH CYAN ACCENT ===================== */
        
        .metric-value {
            font-size: 3rem !important;
            font-weight: 900 !important;
            color: #0891b2;
            margin: 12px 0;
            letter-spacing: -2px;
        }

        .metric-label {
            font-size: 0.7rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .metric-desc {
            color: #94a3b8;
            font-size: 0.875rem;
            margin-top: 8px;
            font-weight: 500;
        }

        /* ===================== CYAN PRIMARY BUTTONS ===================== */
        
        .stButton > button {
            background: #06b6d4 !important;
            border: none !important;
            color: white !important;
            padding: 14px 32px !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 2px 8px rgba(6, 182, 212, 0.25) !important;
            letter-spacing: 0.3px !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.35) !important;
            background: #0891b2 !important;
        }
        
        .stButton > button:active {
            transform: translateY(0px) !important;
        }

        /* ===================== CLEAN SIDEBAR ===================== */
        
        [data-testid="stSidebar"] {
            background: #f8fafc !important;
            border-right: 2px solid #e2e8f0 !important;
        }
        
        [data-testid="stSidebar"] h1 {
            font-size: 1.5rem !important;
            padding: 1rem 0;
            color: #0f172a !important;
        }
        
        [data-testid="stSidebar"] .stRadio > label {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }
        
        [data-testid="stSidebar"] [data-baseweb="radio"] {
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px 16px;
            margin: 6px 0;
            transition: all 0.25s ease;
        }
        
        [data-testid="stSidebar"] [data-baseweb="radio"]:hover {
            background: #e0f2fe;
            border-color: #22d3ee;
        }

        /* ===================== CLEAN INPUTS ===================== */
        
        .stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label {
            color: #334155 !important;
            font-weight: 700 !important;
            font-size: 0.875rem !important;
        }
        
        .stSelectbox > div > div, .stMultiSelect > div > div {
            background: white !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 10px !important;
            transition: all 0.25s ease !important;
        }
        
        .stSelectbox > div > div:hover, .stMultiSelect > div > div:hover {
            border-color: #22d3ee !important;
        }
        
        .stSelectbox > div > div:focus-within, .stMultiSelect > div > div:focus-within {
            border-color: #06b6d4 !important;
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.1) !important;
        }
        
        .stNumberInput input {
            background: white !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 10px !important;
            color: #1e293b !important;
            font-weight: 600 !important;
            padding: 12px !important;
        }
        
        .stNumberInput input:focus {
            border-color: #06b6d4 !important;
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.1) !important;
        }
        
        .stSlider > div > div > div {
            background: #06b6d4 !important;
        }

        /* ===================== CLEAN TABS ===================== */
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #f8fafc;
            border-radius: 12px;
            padding: 6px;
            border: 2px solid #e2e8f0;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: #64748b;
            font-weight: 600;
            padding: 10px 20px;
            transition: all 0.25s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #e0f2fe;
            color: #0891b2;
        }
        
        .stTabs [aria-selected="true"] {
            background: #06b6d4 !important;
            color: white !important;
        }

        /* ===================== CLEAN DATA TABLES ===================== */
        
        [data-testid="stDataFrame"] {
            background: white !important;
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            overflow: hidden !important;
        }
        
        .dataframe {
            font-size: 0.9rem !important;
            color: #334155 !important;
        }
        
        .dataframe thead tr th {
            background: #f1f5f9 !important;
            color: #0f172a !important;
            font-weight: 800 !important;
            padding: 16px !important;
            border-bottom: 2px solid #e2e8f0 !important;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 1px;
        }
        
        .dataframe tbody tr {
            transition: all 0.2s ease;
        }
        
        .dataframe tbody tr:hover {
            background: #f8fafc !important;
        }
        
        .dataframe tbody td {
            padding: 14px !important;
            border-bottom: 1px solid #f1f5f9 !important;
        }

        /* ===================== INFO BOXES ===================== */
        
        .stAlert {
            background: white !important;
            border-radius: 12px !important;
            border: 2px solid #e0f2fe !important;
            border-left: 4px solid #06b6d4 !important;
            padding: 20px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        }
        
        .stAlert p {
            color: #475569 !important;
            font-weight: 500;
        }
        
        .stSuccess {
            background: #f0fdf4 !important;
            border: 2px solid #bbf7d0 !important;
            border-left: 4px solid #22c55e !important;
        }
        
        .stError {
            background: #fef2f2 !important;
            border: 2px solid #fecaca !important;
            border-left: 4px solid #ef4444 !important;
        }
        
        .stWarning {
            background: #fffbeb !important;
            border: 2px solid #fde68a !important;
            border-left: 4px solid #f59e0b !important;
        }

        /* ===================== PLOTLY CHARTS ===================== */
        
        .js-plotly-plot {
            border-radius: 12px !important;
            overflow: hidden !important;
            background: white !important;
            border: 2px solid #e2e8f0;
            padding: 8px;
        }
        
        /* ===================== CUSTOM SCROLLBAR ===================== */
        
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #06b6d4;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #0891b2;
        }

        /* ===================== SMOOTH ANIMATIONS ===================== */
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .neon-card {
            animation: slideIn 0.4s ease-out;
        }
        
        /* ===================== COLUMN SPACING ===================== */
        
        [data-testid="column"] {
            padding: 0 10px;
        }
        
        /* ===================== DIVIDER ===================== */
        
        hr {
            border: none;
            border-top: 2px solid #e2e8f0;
            margin: 2rem 0;
        }
    </style>
    """

def card_component(title, value, description=""):
    return f"""
    <div class="neon-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-desc">{description}</div>
    </div>
    """
