from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import re

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# ── Linear Regression from scratch (CS229 Normal Equation) ──
class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return X_b @ self.theta

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'josaa_data.csv')

print("Loading data...")
df = pd.read_csv(DATA_PATH)

def _normalize_whitespace(s):
    if pd.isna(s):
        return s
    return re.sub(r'\s+', ' ', str(s)).strip()

for _col in ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender']:
    df[_col] = df[_col].apply(_normalize_whitespace)

PAPER2_KEYWORDS = ['Architecture', 'Planning']
paper2_mask = df['Academic Program Name'].str.contains('|'.join(PAPER2_KEYWORDS), case=False, na=False)
df = df[~paper2_mask].reset_index(drop=True)

for col in ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender']:
    df[col] = df[col].astype(str).str.strip()
    
    # 2. Define the list of quotas to be removed entirely
    to_remove = ['GO', 'AP', 'JK', 'LA']
    
    # 3. Create a boolean mask for IITs vs Others
    # This assumes IITs contain "Indian Institute of Technology"
    is_iit = df['Institute'].str.contains('Indian Institute of Technology', case=False, na=False)
    
    # 4. Apply removal for all rows
    df = df[~df['Quota'].isin(to_remove)].copy()
    
    # 5. Apply the conditional merge: 
    # For NITs/IIITs (not IITs), replace 'AI' with 'OS'
    nit_iiit_mask = ~df['Institute'].str.contains('Indian Institute of Technology', case=False, na=False)
    df.loc[nit_iiit_mask & (df['Quota'] == 'AI'), 'Quota'] = 'OS'

df_final = df[df['Round'] == df.groupby('Year')['Round'].transform('max')]
df_clean  = df_final.dropna(
    subset=['Closing Rank', 'Opening Rank', 'Gender', 'Seat Type', 'Quota']
).copy()

df_clean['Seat_Key'] = (
    df_clean['Institute']             + ' | ' +
    df_clean['Academic Program Name'] + ' | ' +
    df_clean['Quota']                 + ' | ' +
    df_clean['Seat Type']             + ' | ' +
    df_clean['Gender']
)

df_model = df_clean.copy()
print(f"Data ready. {df_model['Seat_Key'].nunique()} unique seats loaded.")

def get_inst_type(name):
    n = re.sub(r'\s+', ' ', (name or '').strip()).lower()
    if 'indian institute of technology' in n:                          return 'IIT'
    if 'national institute of technology' in n:                        return 'NIT'
    if 'indian institute of information technology' in n or 'iiit' in n: return 'IIIT'
    return 'GFTI'

# ── Home State Mapping Engine ──
NIT_STATE_MAP = {
    'andhra pradesh': 'AP', 'arunachal pradesh': 'AR', 'silchar': 'AS', 'patna': 'BR',
    'raipur': 'CG', 'delhi': 'DL', 'goa': 'GA', 'surat': 'GJ', 'sardar vallabhbhai': 'GJ',
    'kurukshetra': 'HR', 'hamirpur': 'HP', 'srinagar': 'JK', 'jammu': 'JK', 'jamshedpur': 'JH',
    'surathkal': 'KA', 'calicut': 'KL', 'bhopal': 'MP', 'maulana azad': 'MP', 'nagpur': 'MH',
    'visvesvaraya': 'MH', 'manipur': 'MN', 'meghalaya': 'ML', 'mizoram': 'MZ', 'nagaland': 'NL',
    'rourkela': 'OD', 'puducherry': 'PY', 'jalandhar': 'PB', 'dr. b r ambedkar': 'PB',
    'jaipur': 'RJ', 'malaviya': 'RJ', 'sikkim': 'SK', 'tiruchirappalli': 'TN', 'trichy': 'TN',
    'warangal': 'TG', 'agartala': 'TR', 'allahabad': 'UP', 'motilal nehru': 'UP',
    'uttarakhand': 'UK', 'durgapur': 'WB', 'shibpur': 'WB'
}

def predict_for_student(mains_rank, adv_rank, category, gender, home_state,
                        inst_type_filter='ALL', branch_filter='',
                        predict_year=2026, top_n=50):

    filtered = df_model[
        (df_model['Seat Type'] == category) &
        (df_model['Gender']    == gender)
    ]

    if branch_filter:
        filtered = filtered[
            filtered['Academic Program Name']
            .str.lower()
            .str.contains(branch_filter.lower(), na=False)
        ]

    results = []

    for seat_key, seat_data in filtered.groupby('Seat_Key'):
        parts = seat_key.split(' | ')
        inst_name = parts[0]
        prog_name = parts[1]
        quota     = parts[2]
        inst_type = get_inst_type(inst_name)

        if inst_type_filter != 'ALL' and inst_type != inst_type_filter:
            continue

        # ── Rank & Quota Routing Logic ──
        used_rank = None
        rank_type = ""

        if inst_type == 'IIT':
            if not adv_rank or quota != 'AI': continue
            used_rank = adv_rank
            rank_type = 'Advanced'
            
        elif inst_type == 'IIIT':
            # IIITs have no HS quota — open to all states under OS (renamed from AI)
            if not mains_rank or quota != 'OS': continue
            used_rank = mains_rank
            rank_type = 'Mains'

        elif inst_type == 'GFTI':
            if not mains_rank: continue

            inst_lower = inst_name.lower()
            gfti_state = None
            for key, st in NIT_STATE_MAP.items():
                if key in inst_lower:
                    gfti_state = st
                    break

            is_home = (gfti_state == home_state) if (gfti_state and home_state) else False

            # Same strict HS/OS split as NITs
            if is_home:
                if quota != 'HS': continue
            else:
                if quota != 'OS': continue

            used_rank = mains_rank
            rank_type = 'Mains'
            
        elif inst_type == 'NIT':
            if not mains_rank: continue
            
            inst_lower = inst_name.lower()
            nit_state = None
            for key, st in NIT_STATE_MAP.items():
                if key in inst_lower:
                    nit_state = st
                    break
            
            is_home = (nit_state == home_state) if (nit_state and home_state) else False

            # Strict: home state → HS seats only (student competes in HS pool, not OS)
            #         other state → OS seats only
            # Previously used valid_quotas=['AI','HS'] which could let AI quota
            # seats bleed through before the AI→OS rename ran, and was unclear.
            if is_home:
                if quota != 'HS': continue
            else:
                if quota != 'OS': continue

            used_rank = mains_rank
            rank_type = 'Mains'

        if not used_rank:
            continue

        # ── Regression Logic ──
        seat_data     = seat_data.sort_values('Year')
        closing_ranks = seat_data['Closing Rank'].values.astype(float)
        years         = seat_data['Year'].values.astype(float)
        n_years       = len(closing_ranks)

        hist_mean = float(np.mean(closing_ranks))
        hist_std  = float(np.std(closing_ranks))

        if n_years <= 3:
            predicted_closing = hist_mean
            method = 'average'
        else:
            X      = years.reshape(-1, 1)
            X_mean = X.mean();  X_std = X.std() or 1.0
            X_norm = (X - X_mean) / X_std

            model = LinearRegression()
            model.fit(X_norm, closing_ranks)

            X_fut           = np.array([[(predict_year - X_mean) / X_std]])
            predicted_closing = float(model.predict(X_fut)[0])

            tolerance = 0.05 * hist_mean
            if abs(predicted_closing - hist_mean) > tolerance:
                predicted_closing = round(hist_mean)
                method = 'average'
            else:
                method = 'regression'

        predicted_closing = max(1.0, predicted_closing)

        relative_gap = (predicted_closing - used_rank) / predicted_closing
        PROB_SCALE   = 7.0
        probability  = float(1 / (1 + np.exp(-np.clip(relative_gap * PROB_SCALE, -6, 6)))) * 100

        if n_years <= 3: confidence = 'low'
        elif hist_mean and (hist_std / hist_mean) > 0.20: confidence = 'medium'
        else: confidence = 'high'

        results.append({
            'institute':        inst_name,
            'program':          prog_name,
            'quota':            quota,
            'seat_type':        parts[3],
            'gender':           parts[4],
            'inst_type':        inst_type,
            'predicted_closing': round(predicted_closing),
            'rank_used':        used_rank,
            'rank_type':        rank_type,
            'prob':             round(probability, 1),
            'method':           method,
            'confidence':       confidence,
            'n_years':          n_years
        })

    results.sort(key=lambda x: x['predicted_closing'])

    if top_n != 'ALL':
        results = results[:int(top_n)]

    return results

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        mains_rank = data.get('mains_rank')
        adv_rank = data.get('adv_rank')
        mains_rank = int(mains_rank) if mains_rank else None
        adv_rank = int(adv_rank) if adv_rank else None

        if not mains_rank and not adv_rank:
            return jsonify({'error': 'Please enter at least one rank.'}), 400

        results = predict_for_student(
            mains_rank     = mains_rank,
            adv_rank       = adv_rank,
            category       = data.get('category', 'OPEN'),
            gender         = data.get('gender', 'Gender-Neutral'),
            home_state     = data.get('home_state', ''),
            inst_type_filter = data.get('inst_type', 'ALL'),
            branch_filter  = data.get('branch_filter', ''),
            predict_year   = 2026,
            top_n          = data.get('top_n', 50)
        )
        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)