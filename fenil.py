import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class EnhancedVolatilitySurfacePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.smile_params = {}
        
    def classify_moneyness(self, strike, underlying):
        """Classify option moneyness"""
        ratio = strike / underlying
        if ratio < 0.95:
            return 'deep_otm'
        elif ratio < 0.98:
            return 'otm'
        elif ratio < 1.02:
            return 'atm'
        elif ratio < 1.05:
            return 'itm'
        else:
            return 'deep_itm'
    
    def extract_smile_features_by_region(self, iv_matrix, strikes, underlying_prices):
        """Extract smile features for different moneyness regions"""
        iv_matrix = np.array(iv_matrix, dtype=np.float64)
        iv_matrix = np.where(iv_matrix == -1.0, np.nan, iv_matrix)
        
        features = {
            'atm_level': [], 'atm_slope': [], 'atm_curvature': [],
            'otm_level': [], 'otm_slope': [], 'wing_risk': [],
            'skew': [], 'term_structure': []
        }
        
        for i, row in enumerate(iv_matrix):
            underlying = underlying_prices.iloc[i] if hasattr(underlying_prices, 'iloc') else underlying_prices[i]
            valid_mask = ~np.isnan(row)
            
            if valid_mask.sum() < 3:
                # Fill with default values
                for key in features:
                    features[key].append(np.nan)
                continue
                
            valid_strikes = np.array(strikes)[valid_mask]
            valid_ivs = row[valid_mask]
            
            # ATM region analysis
            atm_mask = np.abs(valid_strikes - underlying) < 500
            if atm_mask.sum() >= 2:
                atm_strikes = valid_strikes[atm_mask]
                atm_ivs = valid_ivs[atm_mask]
                atm_level = np.mean(atm_ivs)
                atm_slope = np.polyfit(atm_strikes - underlying, atm_ivs, 1)[0] if len(atm_ivs) >= 2 else 0
                atm_curvature = np.polyfit(atm_strikes - underlying, atm_ivs, 2)[0] if len(atm_ivs) >= 3 else 0
            else:
                # Interpolate ATM level
                atm_level = np.interp(underlying, valid_strikes, valid_ivs)
                atm_slope = 0
                atm_curvature = 0
            
            # OTM region analysis
            otm_mask = valid_strikes < underlying * 0.98
            if otm_mask.sum() >= 2:
                otm_strikes = valid_strikes[otm_mask]
                otm_ivs = valid_ivs[otm_mask]
                otm_level = np.mean(otm_ivs)
                otm_slope = np.polyfit(otm_strikes - underlying, otm_ivs, 1)[0]
            else:
                otm_level = atm_level * 1.1  # Default higher volatility for OTM
                otm_slope = 0.0001
            
            # Wing risk (volatility at extreme strikes)
            wing_risk = np.std(valid_ivs) if len(valid_ivs) > 1 else 0.05
            
            # Skew measure
            left_wing = valid_ivs[valid_strikes < underlying]
            right_wing = valid_ivs[valid_strikes > underlying]
            skew = (np.mean(left_wing) - np.mean(right_wing)) if len(left_wing) > 0 and len(right_wing) > 0 else 0
            
            # Term structure proxy
            term_structure = atm_level  # Simplified
            
            features['atm_level'].append(atm_level)
            features['atm_slope'].append(atm_slope)
            features['atm_curvature'].append(atm_curvature)
            features['otm_level'].append(otm_level)
            features['otm_slope'].append(otm_slope)
            features['wing_risk'].append(wing_risk)
            features['skew'].append(skew)
            features['term_structure'].append(term_structure)
        
        return pd.DataFrame(features)
    
    def fit_parametric_smile(self, strikes, ivs, underlying):
        """Fit parametric smile model: IV = a + b*log(K/S) + c*log(K/S)^2"""
        # Convert inputs to proper numpy arrays with float64 dtype
        strikes = np.array(strikes, dtype=np.float64)
        ivs = np.array(ivs, dtype=np.float64)
        underlying = float(underlying)
        
        # Handle -1.0 values as NaN
        ivs = np.where(ivs == -1.0, np.nan, ivs)
        
        valid_mask = ~np.isnan(ivs)
        if valid_mask.sum() < 3:
            return {'a': 0.2, 'b': 0, 'c': 0}
        
        valid_strikes = strikes[valid_mask]
        valid_ivs = ivs[valid_mask]
        
        log_moneyness = np.log(valid_strikes / underlying)
        
        try:
            # Fit quadratic in log-moneyness
            coeffs = np.polyfit(log_moneyness, valid_ivs, 2)
            return {'a': coeffs[2], 'b': coeffs[1], 'c': coeffs[0]}
        except:
            return {'a': np.mean(valid_ivs), 'b': 0, 'c': 0}
    
    def parametric_iv(self, strike, underlying, params):
        """Calculate IV using parametric model"""
        log_moneyness = np.log(strike / underlying)
        return params['a'] + params['b'] * log_moneyness + params['c'] * log_moneyness**2
    
    def fit_smile_model(self, train_df):
        """Enhanced model fitting with regional analysis"""
        call_cols = [col for col in train_df.columns if 'call_iv_' in col]
        put_cols = [col for col in train_df.columns if 'put_iv_' in col]
        xi_cols = [col for col in train_df.columns if col.startswith('X')]
        
        call_strikes = [int(col.split('_')[-1]) for col in call_cols]
        put_strikes = [int(col.split('_')[-1]) for col in put_cols]
        
        # Extract enhanced features
        call_feat_df = self.extract_smile_features_by_region(
            train_df[call_cols].values, call_strikes, train_df['underlying']
        )
        put_feat_df = self.extract_smile_features_by_region(
            train_df[put_cols].values, put_strikes, train_df['underlying']
        )
        
        # Fit parametric smile models for each row
        call_params = []
        put_params = []
        
        for i, row in train_df.iterrows():
            # Convert to numpy arrays with proper dtype
            call_ivs = np.array(row[call_cols].values, dtype=np.float64)
            put_ivs = np.array(row[put_cols].values, dtype=np.float64)
            underlying = float(row['underlying'])
            
            call_param = self.fit_parametric_smile(np.array(call_strikes), call_ivs, underlying)
            put_param = self.fit_parametric_smile(np.array(put_strikes), put_ivs, underlying)
            
            call_params.append(call_param)
            put_params.append(put_param)
        
        self.smile_params['call'] = call_params
        self.smile_params['put'] = put_params
        
        # Train ML models for smile parameters
        X = train_df[xi_cols[:25] + ['underlying']].values.astype(np.float64)
        
        for param in ['a', 'b', 'c']:
            for option_type in ['call', 'put']:
                y = [p[param] for p in self.smile_params[option_type]]
                y = np.array(y, dtype=np.float64)
                valid_mask = ~np.isnan(y)
                
                if valid_mask.sum() > 10:
                    self.scalers[f'{option_type}_{param}'] = StandardScaler()
                    X_scaled = self.scalers[f'{option_type}_{param}'].fit_transform(X[valid_mask])
                    
                    self.models[f'{option_type}_{param}'] = RandomForestRegressor(
                        n_estimators=30, max_depth=6, random_state=42, n_jobs=-1
                    )
                    self.models[f'{option_type}_{param}'].fit(X_scaled, y[valid_mask])
    
    def predict_full_smile(self, test_row, option_type='call'):
        """Predict complete volatility smile including missing strikes"""
        xi_cols = [col for col in test_row.index if col.startswith('X')][:25]
        X = test_row[xi_cols + ['underlying']].values.reshape(1, -1).astype(np.float64)
        underlying = float(test_row['underlying'])
        
        # Predict parametric model parameters
        params = {}
        for param in ['a', 'b', 'c']:
            model_key = f'{option_type}_{param}'
            if model_key in self.models:
                X_scaled = self.scalers[model_key].transform(X)
                params[param] = self.models[model_key].predict(X_scaled)[0]
            else:
                # Default values
                if param == 'a':
                    params[param] = 0.2
                else:
                    params[param] = 0
        
        return params
    
    def generate_extended_smile(self, test_row, known_strikes, known_ivs, target_strikes, option_type):
        """Generate smile for extended strike range"""
        underlying = float(test_row['underlying'])
        
        # Get parametric model parameters
        params = self.predict_full_smile(test_row, option_type)
        
        # Calculate IVs for all target strikes
        predicted_ivs = []
        
        for strike in target_strikes:
            if strike in known_strikes:
                strike_idx = known_strikes.index(strike)
                if not pd.isna(known_ivs[strike_idx]):
                    # Use known value
                    predicted_ivs.append(float(known_ivs[strike_idx]))
                    continue
            
            # Predict using parametric model
            iv = self.parametric_iv(strike, underlying, params)
            
            # Apply moneyness-based adjustments
            moneyness = self.classify_moneyness(strike, underlying)
            
            if option_type == 'call':
                if moneyness == 'deep_otm':
                    iv *= 1.2  # Higher vol for deep OTM calls
                elif moneyness == 'deep_itm':
                    iv *= 0.9  # Lower vol for deep ITM calls
            else:  # put
                if moneyness == 'deep_otm':
                    iv *= 1.15  # Higher vol for deep OTM puts
                elif moneyness == 'deep_itm':
                    iv *= 0.95  # Lower vol for deep ITM puts
            
            # Ensure reasonable bounds
            iv = max(0.05, min(2.0, iv))
            predicted_ivs.append(iv)
        
        return predicted_ivs

def main():
    print("Loading data...")
    
    # Load and process data with explicit dtype conversion
    train_df = pd.read_csv('train_data.csv', low_memory=False)
    test_df = pd.read_csv('test_data.csv', low_memory=False)
    
    # Process training data - convert all numeric columns explicitly
    iv_cols = [col for col in train_df.columns if '_iv_' in col]
    xi_cols = [col for col in train_df.columns if col.startswith('X')]
    
    # Convert IV columns to numeric, handling errors
    for col in iv_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        train_df[col] = train_df[col].replace(-1.0, np.nan)
    
    # Convert X columns to numeric
    for col in xi_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    
    # Convert underlying to numeric
    train_df['underlying'] = pd.to_numeric(train_df['underlying'], errors='coerce')
    
    # Process test data
    iv_cols_test = [col for col in test_df.columns if '_iv_' in col]
    xi_cols_test = [col for col in test_df.columns if col.startswith('X')]
    
    for col in iv_cols_test:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    for col in xi_cols_test:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    test_df['underlying'] = pd.to_numeric(test_df['underlying'], errors='coerce')
    
    # Initialize predictor
    predictor = EnhancedVolatilitySurfacePredictor()
    
    print("Training enhanced smile models...")
    predictor.fit_smile_model(train_df)
    
    print("Predicting complete volatility surface...")
    
    # Define complete strike ranges for submission
    call_strikes_full = [23500, 23600, 23700, 23800, 23900, 24000, 24100, 24200, 24300, 24400, 
                        24500, 24600, 24700, 24800, 24900, 25000, 25100, 25200, 25300, 25400, 
                        25500, 25600, 25700, 25800, 25900, 26000, 26100, 26200, 26300, 26400, 26500]
    
    put_strikes_full = [22500, 22600, 22700, 22800, 22900, 23000, 23100, 23200, 23300, 23400, 
                       23500, 23600, 23700, 23800, 23900, 24000, 24100, 24200, 24300, 24400, 
                       24500, 24600, 24700, 24800, 24900, 25000, 25100, 25200, 25300, 25400, 25500]
    
    # Available strikes in test data
    call_cols_test = [col for col in test_df.columns if 'call_iv_' in col]
    put_cols_test = [col for col in test_df.columns if 'put_iv_' in col]
    
    call_strikes_test = [int(col.split('_')[-1]) for col in call_cols_test]
    put_strikes_test = [int(col.split('_')[-1]) for col in put_cols_test]
    
    results = []
    
    for idx, row in test_df.iterrows():
        # Get known values and convert to float
        call_ivs_known = [float(x) if not pd.isna(x) else np.nan for x in row[call_cols_test].values]
        put_ivs_known = [float(x) if not pd.isna(x) else np.nan for x in row[put_cols_test].values]
        
        # Predict complete call smile
        call_ivs_full = predictor.generate_extended_smile(
            row, call_strikes_test, call_ivs_known, call_strikes_full, 'call'
        )
        
        # Predict complete put smile
        put_ivs_full = predictor.generate_extended_smile(
            row, put_strikes_test, put_ivs_known, put_strikes_full, 'put'
        )
        
        # Create result row
        result_row = {'timestamp': row['timestamp']}
        
        # Add call IVs
        for i, strike in enumerate(call_strikes_full):
            result_row[f'call_iv_{strike}'] = call_ivs_full[i]
        
        # Add put IVs
        for i, strike in enumerate(put_strikes_full):
            result_row[f'put_iv_{strike}'] = put_ivs_full[i]
        
        results.append(result_row)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(results)
    
    # Save results
    submission_df.to_csv('sub_new.csv', index=False)
    print("Enhanced results saved to sub_new.csv")
    print(f"Generated IVs for {len(call_strikes_full)} call strikes and {len(put_strikes_full)} put strikes")

if __name__ == "__main__":
    main()
