import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from datetime import timedelta

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Analytics de Engagement & Riesgo de Attrition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%); font-family: 'Inter', sans-serif; }
    h1 { color: #1e293b; font-weight: 700; font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h2, h3 { color: #334155; font-weight: 600; }
    .stMetric { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border-left: 4px solid #667eea; }
    .stMetric:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    .alert-box { padding: 16px; border-radius: 12px; margin: 10px 0; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .alert-critical { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-left: 4px solid #dc2626; color: #991b1b; }
    .alert-warning { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 4px solid #f59e0b; color: #92400e; }
    .alert-success { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-left: 4px solid #10b981; color: #065f46; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #334155 100%); }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FUNCIONES DE CARGA Y GENERACIÓN DE DATOS
# -----------------------------------------------------------------------------
@st.cache_data
def generate_mock_data():
    """Genera datos ficticios robustos para pruebas."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-06-30", freq='D')
    teams = ['Breaking Badger', "Finding Nemo's Friends", 'Alpha Squad', 'Data Wizards', 'Phoenix Team']
    locations = ['MX/JALISCO/GDL', 'CO/ANT/MED', 'AR/BA/CABA', 'MX/CDMX/POL']
    
    data_list = []
    
    for i in range(80): 
        email = f"employee_{i}@company.com"
        team = np.random.choice(teams)
        loc = np.random.choice(locations)
        seniority = np.random.choice(['Junior', 'Semi-Senior', 'Senior'])
        
        # Base engagement por equipo
        if team == 'Breaking Badger': base_eng = np.random.uniform(1.5, 2.5)
        elif team == "Finding Nemo's Friends": base_eng = np.random.uniform(2.0, 3.0)
        else: base_eng = np.random.uniform(3.0, 4.5)
        
        for d in dates:
            daily_noise = np.random.normal(0, 0.2)
            current_eng = np.clip(base_eng + daily_noise, 1, 5)
            
            if np.random.random() < 0.01:
                base_eng = max(1, base_eng - 1.0)
            
            if base_eng < 3.0:
                base_eng += 0.01

            data_list.append({
                'Date': d, 'Email': email, 'Team Name': team, 
                'Location': loc, 'Seniority': seniority, 
                'Engagement': round(current_eng, 2)
            })
            
    return pd.DataFrame(data_list)

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return generate_mock_data()
    else:
        return generate_mock_data()

# -----------------------------------------------------------------------------
# LÓGICA DE NEGOCIO (REPRODUCIBLE)
# -----------------------------------------------------------------------------

def discretize_data(df, method='quartiles'):
    """Discretización basada en lógica de referencia."""
    df_copy = df.copy()
    df_copy = df_copy.drop_duplicates()
    
    if method == 'quartiles':
        q = df_copy['Engagement'].quantile([0.25, 0.5, 0.75])
        def cat_quartiles_num(x):
            if x <= q[0.25]: return 0
            elif x <= q[0.5]: return 1
            elif x <= q[0.75]: return 2
            else: return 3
        df_copy['State_Num'] = df_copy['Engagement'].apply(cat_quartiles_num)
        labels = {0: 'Bajo', 1: 'Medio-Bajo', 2: 'Medio-Alto', 3: 'Alto'}
        n_states = 4
        
    elif method == 'terciles':
        t = df_copy['Engagement'].quantile([0.33, 0.66])
        def cat_terciles_num(x):
            if x <= t[0.33]: return 0
            elif x <= t[0.66]: return 1
            else: return 2
        df_copy['State_Num'] = df_copy['Engagement'].apply(cat_terciles_num)
        labels = {0: 'Bajo', 1: 'Medio', 2: 'Alto'}
        n_states = 3
        
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(df_copy[['Engagement']])
        centers = kmeans.cluster_centers_.flatten()
        sorted_idx = np.argsort(centers)
        map_labels = {original: new for new, original in enumerate(sorted_idx)}
        cluster_temp = kmeans.predict(df_copy[['Engagement']])
        df_copy['State_Num'] = pd.Series(cluster_temp).map(map_labels).values
        labels = {0: 'Cluster 0 (Bajo)', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3 (Alto)'}
        n_states = 4
        
    return df_copy, labels, n_states

def calculate_transition_matrix(df, group_col, target_group, state_col, n_states):
    """Calcula matriz de transición replicando lógica de agregación del notebook."""
    subset = df[df[group_col] == target_group].sort_values('Date')
    
    if subset.empty:
        return np.zeros((n_states, n_states)), np.zeros(n_states)

    daily = subset.groupby('Date')[state_col].mean().round().astype(int).reset_index()
    states = daily[state_col].values
    
    if len(states) < 2:
        return np.zeros((n_states, n_states)), np.zeros(n_states)

    transitions = {}
    for s1, s2 in zip(states[:-1], states[1:]):
        if s1 not in transitions:
            transitions[s1] = {}
        if s2 not in transitions[s1]:
            transitions[s1][s2] = 0
        transitions[s1][s2] += 1
    
    matrix = np.zeros((n_states, n_states))
    for s1 in range(n_states):
        dests = transitions.get(s1, {})
        total = sum(dests.values())
        if total > 0:
            for s2 in range(n_states):
                count = dests.get(s2, 0)
                matrix[s1, s2] = count / total
            
    return matrix, matrix.sum(axis=1)

def detect_low_engagement_groups(df, analysis_level, threshold=2.5):
    """Detecta grupos con engagement promedio bajo."""
    group_avg = df.groupby(analysis_level)['Engagement'].agg(['mean', 'std', 'count']).reset_index()
    group_avg.columns = ['Group', 'Avg_Engagement', 'Std_Dev', 'Records']
    critical = group_avg[group_avg['Avg_Engagement'] < threshold * 0.8]
    warning = group_avg[(group_avg['Avg_Engagement'] >= threshold * 0.8) & (group_avg['Avg_Engagement'] < threshold)]
    return critical, warning, group_avg

# -----------------------------------------------------------------------------
# INTERFAZ DE USUARIO - SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Configuración</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Cargar CSV", type=['csv'], help="Formato: Date, Email, Team Name, Location, Seniority, Engagement")
    
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    method_selector = st.selectbox(
        "🔬 Método de Discretización",
        options=['quartiles', 'terciles', 'kmeans'],
        format_func=lambda x: {'quartiles': 'Cuartiles (4 estados)', 'terciles': 'Terciles (3 estados)', 'kmeans': 'K-Means (4 clusters)'}[x]
    )
    
    analysis_level = st.radio(
        "Nivel de Análisis",
        options=['Team Name', 'Location'],
        format_func=lambda x: f"{'Equipos' if x == 'Team Name' else 'Ubicaciones'}"
    )
    
    engagement_threshold = st.slider(
        "⚠️ Umbral de Alerta de Engagement",
        min_value=1.0, max_value=5.0, value=3.0, step=0.1
    )
    
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    st.info("💡 Tip: Se recomienda usar el modelo de **Terciles y Ubicación** pues es el más apegado a la realidad. ")

# -----------------------------------------------------------------------------
# PROCESAMIENTO
# -----------------------------------------------------------------------------
raw_data = load_data(uploaded_file)
df_processed, state_labels, num_states = discretize_data(raw_data, method=method_selector)
critical_groups, warning_groups, all_groups = detect_low_engagement_groups(raw_data, analysis_level, engagement_threshold)

# -----------------------------------------------------------------------------
# DASHBOARD PRINCIPAL
# -----------------------------------------------------------------------------
st.title("Monitor de Engagement & Riesgo de Attrition")

if len(critical_groups) > 0 or len(warning_groups) > 0:
    st.markdown("### 🚨 Alertas de Engagement")
    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        if len(critical_groups) > 0:
            st.markdown(f"""
            <div class='alert-box alert-critical'>
                <strong>⛔ RIESGO CRÍTICO</strong><br/>
                {len(critical_groups)} {analysis_level.lower()}(s) con engagement < {engagement_threshold * 0.8:.1f}:<br/>
                <ul style='margin-top: 8px;'>{''.join([f"<li><strong>{row['Group']}</strong>: {row['Avg_Engagement']:.2f}</li>" for _, row in critical_groups.iterrows()])}</ul>
            </div>
            """, unsafe_allow_html=True)
        else: st.success("✅ Sin Alertas Críticas")
    with alert_col2:
        if len(warning_groups) > 0:
            st.markdown(f"""
            <div class='alert-box alert-warning'>
                <strong>⚠️ PRECAUCIÓN</strong><br/>
                {len(warning_groups)} {analysis_level.lower()}(s) con engagement bajo:<br/>
                <ul style='margin-top: 8px;'>{''.join([f"<li><strong>{row['Group']}</strong>: {row['Avg_Engagement']:.2f}</li>" for _, row in warning_groups.iterrows()])}</ul>
            </div>
            """, unsafe_allow_html=True)
        else: st.success("✅ Sin Alertas de Precaución")

st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

# --- MÉTRICAS PRINCIPALES ---
col1, col2, col3, col4, col5 = st.columns(5)

curr_engagement = raw_data['Engagement'].mean()
last_week = raw_data['Date'].max() - timedelta(days=7)
prev_engagement = raw_data[raw_data['Date'] < last_week]['Engagement'].mean()
diff = curr_engagement - prev_engagement

with col1: st.metric("Engagement Promedio", f"{curr_engagement:.2f}", f"{diff:+.2f}")

# --- CAMBIO APLICADO AQUÍ ---
with col2:
    # Agrupamos por Email y calculamos el promedio histórico de cada persona
    avg_engagement_per_person = raw_data.groupby('Email')['Engagement'].mean()
    # Contamos cuántas personas tienen un promedio menor al umbral del slider
    risk_count = (avg_engagement_per_person < engagement_threshold).sum()
    
    st.metric(f"Personas en Riesgo (< {engagement_threshold})", f"{risk_count}")
# ----------------------------

with col3: st.metric("Total Empleados", f"{raw_data['Email'].nunique()}")
with col4: st.metric(f"🏢 {analysis_level}", f"{raw_data[analysis_level].nunique()}")
with col5:
    healthy_count = len(all_groups[all_groups['Avg_Engagement'] >= engagement_threshold])
    st.metric("Grupos Saludables", f"{healthy_count}/{len(all_groups)}")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Panorama General", "Análisis de Transición (Markov)", "Comparativa de Riesgo"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Distribución de Engagement")
        fig_hist = px.histogram(raw_data, x="Engagement", nbins=40, color_discrete_sequence=['#667eea'], marginal="violin")
        fig_hist.add_vline(x=engagement_threshold, line_dash="dash", line_color="red")
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.markdown("#### Evolución Temporal")
        daily_avg = raw_data.groupby('Date')['Engagement'].mean().reset_index()
        fig_line = px.line(daily_avg, x='Date', y='Engagement', markers=True, color_discrete_sequence=['#764ba2'])
        fig_line.add_hline(y=engagement_threshold, line_dash="dash", line_color="red")
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", hovermode='x unified')
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("#### Engagement por Grupo")
    fig_box = px.box(raw_data, x=analysis_level, y='Engagement', color=analysis_level, points="outliers")
    fig_box.add_hline(y=engagement_threshold, line_dash="dash", line_color="red")
    fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.markdown("### Dinámica de Transición de Estados")
    target_list = sorted(df_processed[analysis_level].dropna().unique())
    selected_target = st.selectbox(f"Selecciona {analysis_level}:", target_list, key="markov_selector")
    
    trans_matrix, _ = calculate_transition_matrix(df_processed, analysis_level, selected_target, 'State_Num', num_states)
    
    cm1, cm2 = st.columns([2.5, 1.5])
    with cm1:
        axis_labels = [state_labels[i] for i in range(num_states)]
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=trans_matrix, x=axis_labels, y=axis_labels,
            colorscale='RdYlGn', reversescale=True,
            text=np.round(trans_matrix, 3), texttemplate="<b>%{text}</b>",
            textfont={"size": 14}, showscale=True, zmin=0, zmax=1
        ))
        fig_heatmap.update_layout(title=f"Matriz de Transición: {selected_target}", xaxis_title="Estado Siguiente (t+1)", yaxis_title="Estado Actual (t)", height=500, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    with cm2:
        st.markdown("#### Diagnóstico de Riesgo")
        p_stuck = trans_matrix[0, 0]
        p_fall = trans_matrix[1, 0] if num_states > 1 else 0
        p_rec = trans_matrix[0, 1] if num_states > 1 else 0
        st.metric("Inercia Negativa", f"{p_stuck*100:.1f}%")
        st.metric("Riesgo de Caída", f"{p_fall*100:.1f}%")
        st.metric("Capacidad de Recuperación", f"{p_rec*100:.1f}%")
        
        risk_score = max(0, min(1, (p_stuck * 0.5) + (p_fall * 0.3) - (p_rec * 0.2)))
        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
        if risk_score > 0.6: st.error(f"🚨 ALERTA ROJA (Score: {risk_score:.2f})")
        elif risk_score > 0.35: st.warning(f"⚠️ ALERTA AMARILLA (Score: {risk_score:.2f})")
        else: st.success(f"✅ ESTABLE (Score: {risk_score:.2f})")

with tab3:
    st.markdown("### Ranking de Riesgo por Grupo")
    risk_data = []
    for entity in target_list:
        tm, _ = calculate_transition_matrix(df_processed, analysis_level, entity, 'State_Num', num_states)
        ir, dr, rec = tm[0, 0], (tm[1, 0] if num_states > 1 else 0), (tm[0, 1] if num_states > 1 else 0)
        tr = max(0, min(1, (ir * 0.5) + (dr * 0.3) - (rec * 0.2)))
        avg_eng = raw_data[raw_data[analysis_level] == entity]['Engagement'].mean()
        risk_data.append({'Entity': entity, 'Engagement Promedio': round(avg_eng, 2), 'Inercia Bajo': round(ir, 3), 'Risk Index': round(tr, 3)})
        
    df_risk = pd.DataFrame(risk_data).sort_values('Risk Index', ascending=False)
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Bar(name='Risk Index', x=df_risk['Entity'], y=df_risk['Risk Index'], marker_color=df_risk['Risk Index'].apply(lambda x: '#dc2626' if x > 0.6 else '#f59e0b' if x > 0.35 else '#10b981'), text=df_risk['Risk Index'], textposition='outside'))
    fig_risk.update_layout(title="Ranking de Riesgo", yaxis_title="Risk Index", height=400, showlegend=False)
    st.plotly_chart(fig_risk, use_container_width=True)
    st.dataframe(df_risk.style.background_gradient(cmap='RdYlGn_r', subset=['Risk Index']), use_container_width=True)