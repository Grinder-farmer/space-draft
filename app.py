import streamlit as st
import numpy as np
from skyfield.api import load, wgs84
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
from datetime import datetime as dt_datetime, timezone
import itertools

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Sat Simulator", layout="wide")
st.title("🛰️ Симулятор ориентации спутника")

# --- КЭШИРОВАНИЕ ЗАГРУЗКИ SKYFIELD ---
@st.cache_resource
def load_skyfield_data():
    ts = load.timescale()
    eph = load('de421.bsp')
    return ts, eph

ts, eph = load_skyfield_data()
earth = eph['earth']
sun = eph['sun']

# --- МАТЕМАТИКА ---
def solve_sun_vector_body(I_sb, I_max, n_panels):
    s_sun_un = np.zeros(3)
    active_panels = 0
    for i, n_vec in n_panels.items():
        if (max_i := I_max.get(i, 1.0)) > 1e-6:
            cos_alpha = np.clip(I_sb.get(i, 0.0) / max_i, 0.0, 1.0)
            if cos_alpha > 1e-3:
                s_sun_un += n_vec * cos_alpha
                active_panels += 1
    if active_panels == 0:
        return None
    norm = np.linalg.norm(s_sun_un)
    return s_sun_un / norm if norm > 1e-9 else None

def get_global_vectors(lat, lon, alt, time_obj):
    observer = earth + wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon, elevation_m=alt * 1000)
    sat_pos_gcrs = observer.at(time_obj).position.km
    sun_pos_gcrs = sun.at(time_obj).position.km
    earth_pos_gcrs = earth.at(time_obj).position.km
    
    vec_sat_to_sun = sun_pos_gcrs - sat_pos_gcrs
    s_sun_global = vec_sat_to_sun / np.linalg.norm(vec_sat_to_sun)
    
    vec_sat_to_earth = earth_pos_gcrs - sat_pos_gcrs
    nadir_global = vec_sat_to_earth / np.linalg.norm(vec_sat_to_earth)
    return s_sun_global, nadir_global

def calculate_attitude_triad_final(v1_b, v2_b, v1_g, v2_g):
    t1_b = v1_b / np.linalg.norm(v1_b)
    t2_b_un = np.cross(t1_b, v2_b)
    if np.linalg.norm(t2_b_un) < 1e-9: raise ValueError("Векторы коллинеарны.")
    t2_b = t2_b_un / np.linalg.norm(t2_b_un)
    t3_b = np.cross(t1_b, t2_b)
    M_body = np.array([t1_b, t2_b, t3_b]).T
    
    t1_g = v1_g / np.linalg.norm(v1_g)
    t2_g_un = np.cross(t1_g, v2_g)
    if np.linalg.norm(t2_g_un) < 1e-9: raise ValueError("Векторы коллинеарны.")
    t2_g = t2_g_un / np.linalg.norm(t2_g_un)
    t3_g = np.cross(t1_g, t2_g)
    M_global = np.array([t1_g, t2_g, t3_g]).T
    
    return M_global @ M_body.T

def find_coords_for_sun_angle(target_angle_deg, time_obj, alt_km=550):
    target_cos = np.cos(np.radians(target_angle_deg))
    best_coords = (0, 0)
    min_diff = float('inf')
    for lat in range(-90, 91, 15):
        for lon in range(-180, 181, 15):
            s, n = get_global_vectors(lat, lon, alt_km, time_obj)
            current_cos = np.dot(s, n)
            diff = abs(current_cos - target_cos)
            if diff < min_diff:
                min_diff = diff
                best_coords = (lat, lon)
    return best_coords[0], best_coords[1], alt_km

def create_cube_mesh(center, dims, rotation_matrix, color, opacity=1.0, name='Body'):
    dx, dy, dz = dims[0]/2, dims[1]/2, dims[2]/2
    vertices = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy, dz],  [dx, -dy, dz],  [dx, dy, dz],  [-dx, dy, dz]
    ])
    rotated_vertices = (rotation_matrix @ vertices.T).T
    final_vertices = rotated_vertices + center
    x, y, z = final_vertices[:,0], final_vertices[:,1], final_vertices[:,2]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [2, 3, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [1, 2, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, flatshading=True, name=name, showscale=False)

# --- ИНТЕРФЕЙС ---
st.sidebar.header("Параметры симуляции")

use_current_time = st.sidebar.checkbox("Использовать текущее время", value=True)
if use_current_time:
    time_utc = ts.now()
else:
    d = st.sidebar.date_input("Дата", dt_datetime.now())
    t = st.sidebar.time_input("Время", dt_datetime.now().time())
    dt_val = dt_datetime.combine(d, t).replace(tzinfo=timezone.utc)
    time_utc = ts.from_datetime(dt_val)

target_angle = st.sidebar.slider("Желаемый угол Солнце-Надир (°)", 0.0, 180.0, 90.0)

st.sidebar.subheader("Токи солнечных батарей (А)")
i1 = st.sidebar.number_input("Ток СБ №1", value=0.14, step=0.01)
i2 = st.sidebar.number_input("Ток СБ №2", value=0.04, step=0.01)
i3 = st.sidebar.number_input("Ток СБ №3", value=0.04, step=0.01)

I_sb_current = {1: i1, 2: i2, 3: i3}
I_max_panel = {1: 0.15, 2: 0.15, 3: 0.15}
SAT_DIMS = np.array([0.1, 0.1, 0.2])

if 'calculated' not in st.session_state:
    st.session_state['calculated'] = False
if 'hypotheses' not in st.session_state:
    st.session_state['hypotheses'] = []

if st.sidebar.button("Запустить симуляцию"):
    st.session_state['calculated'] = True
    with st.spinner('Вычисляем варианты ориентации...'):
        sat_lat, sat_lon, sat_alt = find_coords_for_sun_angle(target_angle, time_utc)
        s_sun_global, nadir_global = get_global_vectors(sat_lat, sat_lon, sat_alt, time_utc)
        
        st.session_state['global_vectors'] = (s_sun_global, nadir_global)
        
        found_hypotheses = []
        possible_axes = {"+X":np.array([1,0,0]),"-X":np.array([-1,0,0]),
                         "+Y":np.array([0,1,0]),"-Y":np.array([0,-1,0]),
                         "+Z":np.array([0,0,1]),"-Z":np.array([0,0,-1])}
        
        panel_ids = sorted(list(I_sb_current.keys()))
        face_names = list(possible_axes.keys())

        for nadir_axis_name, nadir_axis_vec in possible_axes.items():
            for panel_face_indices in itertools.permutations(range(len(face_names)), len(panel_ids)):
                
                panel_config_hyp = {pid: possible_axes[face_names[face_idx]] for pid, face_idx in zip(panel_ids, panel_face_indices)}
                
                impossible_physics = False
                current_threshold = 0.005 
                
                active_panels_list = []
                for pid, vec in panel_config_hyp.items():
                    if I_sb_current[pid] > current_threshold:
                        active_panels_list.append(vec)
                
                for v1 in active_panels_list:
                    for v2 in active_panels_list:
                        if np.dot(v1, v2) < -0.9: 
                            impossible_physics = True
                            break
                    if impossible_physics: break
                
                if impossible_physics:
                    continue 

                try:
                    s_sun_body_hyp = solve_sun_vector_body(I_sb_current, I_max_panel, panel_config_hyp)
                    if s_sun_body_hyp is None: continue
                    attitude_matrix_hyp = calculate_attitude_triad_final(nadir_axis_vec, s_sun_body_hyp, nadir_global, s_sun_global)
                    s_sun_global_in_body = attitude_matrix_hyp.T @ s_sun_global
                    consistency_score = np.dot(s_sun_body_hyp, s_sun_global_in_body)
                    
                    if consistency_score > 0.98:
                        config_desc = f"Надир:{nadir_axis_name} | Панели:" + ",".join([f"{face_names[idx]}" for idx in panel_face_indices])
                        
                        found_hypotheses.append({
                            'score': consistency_score,
                            'attitude': attitude_matrix_hyp,
                            'nadir_axis': nadir_axis_name,
                            'panel_config': panel_config_hyp,
                            'desc_full': config_desc,
                            'desc_short': f"Надир: {nadir_axis_name} | Панели: " + ", ".join([face_names[idx] for idx in panel_face_indices])
                        })
                except ValueError:
                    continue
        
        unique_hypotheses = []
        seen_configs = set()
        for h in sorted(found_hypotheses, key=lambda x: x['score'], reverse=True):
            if h['desc_full'] not in seen_configs:
                unique_hypotheses.append(h)
                seen_configs.add(h['desc_full'])
            if len(unique_hypotheses) >= 15: break 
            
        st.session_state['hypotheses'] = unique_hypotheses

if st.session_state['calculated']:
    hypotheses = st.session_state['hypotheses']
    
    if not hypotheses:
        st.error("Не удалось найти физически возможную ориентацию (проверьте токи).")
    else:
        st.info(f"Найдено {len(hypotheses)} возможных вариантов.")
        options = [f"{h['desc_short']} (Score: {h['score']:.4f})" for i, h in enumerate(hypotheses)]
        selected_option_str = st.selectbox("Выберите вариант:", options)
        
        selected_index = options.index(selected_option_str)
        best_hyp = hypotheses[selected_index]
        s_sun_global, nadir_global = st.session_state['global_vectors']

        attitude_matrix = best_hyp['attitude']
        panel_config = best_hyp['panel_config']
        
        SCENE_DOWN_VECTOR = np.array([0.0, 0.0, -1.0])
        scene_alignment_rotation, _ = R.align_vectors(SCENE_DOWN_VECTOR, nadir_global)
        scene_alignment_matrix = scene_alignment_rotation.as_matrix()
        final_transform_matrix = scene_alignment_matrix @ attitude_matrix
        sun_vector_in_scene = scene_alignment_matrix @ s_sun_global
        
        fig = go.Figure()
        fig.add_trace(create_cube_mesh(np.array([0,0,0]), SAT_DIMS, final_transform_matrix, 'firebrick', name='Спутник'))
        
        margin, thickness = 0.95, 0.005
        for _, normal_vec in panel_config.items():
            axis_idx = np.where(np.abs(normal_vec) > 0.5)[0][0]
            p_dims = SAT_DIMS.copy()
            p_dims[axis_idx] = thickness
            mask = np.ones(3, dtype=bool); mask[axis_idx] = False
            p_dims[mask] *= margin
            offset_vec = normal_vec * (SAT_DIMS[axis_idx] / 2)
            center_pos = final_transform_matrix @ offset_vec
            fig.add_trace(create_cube_mesh(center_pos, p_dims, final_transform_matrix, 'deepskyblue', name='Панель'))

        axis_len = np.max(SAT_DIMS) * 2.0
        colors, labels = ['red', 'green', 'blue'], ['+X', '+Y', '+Z']
        for i in range(3):
            vec = np.zeros(3); vec[i] = 1.0
            transformed_vec = final_transform_matrix @ vec * axis_len
            fig.add_trace(go.Scatter3d(x=[0, transformed_vec[0]], y=[0, transformed_vec[1]], z=[0, transformed_vec[2]], mode='lines+text', line=dict(color=colors[i], width=5), text=["", labels[i]], showlegend=False))

        sun_vec_scaled = sun_vector_in_scene * axis_len * 1.5
        fig.add_trace(go.Scatter3d(x=[0, sun_vec_scaled[0]], y=[0, sun_vec_scaled[1]], z=[0, sun_vec_scaled[2]], mode='lines+text', line=dict(color='yellow', width=6), text=["", "SUN"], name="SUN"))
        
        earth_dist = np.max(SAT_DIMS) * 8
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[-earth_dist], mode='markers+text', marker=dict(size=40, color='green', symbol='circle'), text=["EARTH"], textposition="bottom center", name="Earth"))
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, -earth_dist], mode='lines', line=dict(color='white', width=2, dash='dash'), showlegend=False))

        max_range = np.max(SAT_DIMS) * 2
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='black', aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0), height=700)
        st.plotly_chart(fig, use_container_width=True)