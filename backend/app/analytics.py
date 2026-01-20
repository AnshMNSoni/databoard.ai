import pandas as pd
import numpy as np

def analyze_data(df: pd.DataFrame, fields):
    """
    Analyze Aadhaar enrollment data for selected fields.
    Returns meaningful insights instead of basic statistics.
    """
    result = {}
    
    # If analyzing all fields, provide comprehensive summary
    if len(fields) == len(df.columns) or not fields:
        result['summary'] = _get_comprehensive_summary(df)
    
    for field in fields:
        if field not in df.columns:
            continue
            
        if field == 'state':
            result[field] = _analyze_state(df)
        elif field == 'district':
            result[field] = _analyze_district(df)
        elif field == 'date':
            result[field] = _analyze_date(df)
        elif field in ['age_0_5', 'age_5_17', 'age_18_greater']:
            result[field] = _analyze_age_group(df, field)
        elif field == 'pincode':
            result[field] = _analyze_pincode(df)
        elif pd.api.types.is_numeric_dtype(df[field]):
            result[field] = _analyze_numeric(df, field)
        else:
            result[field] = _analyze_categorical(df, field)
    
    return result


def _get_comprehensive_summary(df):
    """Get overall dataset summary"""
    total_enrollments = 0
    if 'age_0_5' in df.columns:
        total_enrollments = (
            df['age_0_5'].fillna(0).sum() + 
            df['age_5_17'].fillna(0).sum() + 
            df['age_18_greater'].fillna(0).sum()
        )
    
    return {
        'total_records': len(df),
        'total_enrollments': int(total_enrollments),
        'states_covered': df['state'].nunique() if 'state' in df.columns else 0,
        'districts_covered': df['district'].nunique() if 'district' in df.columns else 0,
        'date_range': {
            'start': str(df['date'].min()) if 'date' in df.columns else 'N/A',
            'end': str(df['date'].max()) if 'date' in df.columns else 'N/A'
        }
    }


def _analyze_state(df):
    """Analyze state-wise enrollment"""
    if 'age_0_5' not in df.columns:
        return df['state'].value_counts().head(10).to_dict()
    
    state_data = df.groupby('state').agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum', 
        'age_18_greater': 'sum'
    }).reset_index()
    
    state_data['total'] = state_data['age_0_5'] + state_data['age_5_17'] + state_data['age_18_greater']
    state_data = state_data.sort_values('total', ascending=False)
    
    top_states = state_data.head(10).to_dict('records')
    bottom_states = state_data.tail(5).to_dict('records')
    
    return {
        'top_states': [{'name': s['state'], 'value': int(s['total'])} for s in top_states],
        'bottom_states': [{'name': s['state'], 'value': int(s['total'])} for s in bottom_states],
        'total_states': len(state_data),
        'insight': f"Top performer: {top_states[0]['state']} with {int(top_states[0]['total']):,} enrollments"
    }


def _analyze_district(df):
    """Analyze district-wise enrollment"""
    district_counts = df['district'].value_counts().head(15)
    
    return {
        'top_districts': [{'name': str(k), 'value': int(v)} for k, v in district_counts.items()],
        'total_districts': df['district'].nunique(),
        'insight': f"Most active district: {district_counts.index[0]} with {district_counts.iloc[0]:,} records"
    }


def _analyze_date(df):
    """Analyze enrollment trends over time"""
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
    
    monthly = df_copy.groupby('month').size().reset_index(name='count')
    monthly = monthly.sort_values('month').tail(12)
    
    return {
        'monthly_trend': [{'name': r['month'], 'value': int(r['count'])} for _, r in monthly.iterrows()],
        'date_range': {
            'start': str(df_copy['date'].min())[:10],
            'end': str(df_copy['date'].max())[:10]
        },
        'insight': f"Data spans from {str(df_copy['date'].min())[:10]} to {str(df_copy['date'].max())[:10]}"
    }


def _analyze_age_group(df, field):
    """Analyze specific age group enrollment"""
    total = df[field].fillna(0).sum()
    
    # Get state-wise breakdown for this age group
    if 'state' in df.columns:
        state_breakdown = df.groupby('state')[field].sum().sort_values(ascending=False).head(10)
        top_states = [{'name': str(k), 'value': int(v)} for k, v in state_breakdown.items()]
    else:
        top_states = []
    
    age_labels = {
        'age_0_5': '0-5 years (Infants & Toddlers)',
        'age_5_17': '5-17 years (Children & Adolescents)',
        'age_18_greater': '18+ years (Adults)'
    }
    
    return {
        'total_enrollments': int(total),
        'age_group': age_labels.get(field, field),
        'top_states': top_states,
        'average_per_record': round(df[field].fillna(0).mean(), 2),
        'insight': f"{age_labels.get(field, field)}: {int(total):,} total enrollments"
    }


def _analyze_pincode(df):
    """Analyze pincode distribution"""
    pincode_counts = df['pincode'].value_counts().head(10)
    
    return {
        'top_pincodes': [{'name': str(k), 'value': int(v)} for k, v in pincode_counts.items()],
        'total_pincodes': df['pincode'].nunique(),
        'insight': f"Data covers {df['pincode'].nunique():,} unique pincodes"
    }


def _analyze_numeric(df, field):
    """Analyze any numeric field"""
    return {
        'total': int(df[field].fillna(0).sum()),
        'average': round(df[field].fillna(0).mean(), 2),
        'records_with_value': int((df[field] > 0).sum()),
        'insight': f"Total {field}: {int(df[field].fillna(0).sum()):,}"
    }


def _analyze_categorical(df, field):
    """Analyze any categorical field"""
    value_counts = df[field].value_counts().head(10)
    
    return {
        'top_values': [{'name': str(k), 'value': int(v)} for k, v in value_counts.items()],
        'unique_count': df[field].nunique(),
        'insight': f"Most common {field}: {value_counts.index[0]} ({value_counts.iloc[0]:,} records)"
    }
