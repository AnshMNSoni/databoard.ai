import pandas as pd
from app.data_store import data_store
from app.ml_service import ml_service

def suggest_charts(fields):
    """
    Generate AI-powered chart suggestions based on data and ML insights.
    """
    df = data_store.df
    suggestions = []

    if df is None or df.empty:
        return []

    # Get ML results for enhanced suggestions
    ml_results = ml_service.get_results()

    # If no fields provided, generate default Aadhaar-focused charts
    if not fields:
        suggestions = _generate_default_suggestions(df, ml_results)
        return suggestions

    for field in fields:
        if field not in df.columns:
            continue

        suggestion = _generate_field_suggestion(df, field, ml_results)
        if suggestion:
            suggestions.append(suggestion)

    return suggestions


def _generate_default_suggestions(df, ml_results):
    """Generate default chart suggestions for Aadhaar data"""
    suggestions = []
    
    # Chart 1: State-wise enrollment
    if 'state' in df.columns and 'age_0_5' in df.columns:
        state_totals = df.groupby('state').agg({
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).sum(axis=1).sort_values(ascending=False).head(10)
        
        suggestions.append({
            'field': 'state',
            'title': 'Top 10 States by Enrollment',
            'recommended_chart': 'bar',
            'data': [{'name': str(k), 'value': int(v)} for k, v in state_totals.items()],
            'insight': f"Top state: {state_totals.index[0]} with {int(state_totals.iloc[0]):,} enrollments"
        })
    
    # Chart 2: Age group distribution
    if all(col in df.columns for col in ['age_0_5', 'age_5_17', 'age_18_greater']):
        age_totals = {
            '0-5 Years': int(df['age_0_5'].fillna(0).sum()),
            '5-17 Years': int(df['age_5_17'].fillna(0).sum()),
            '18+ Years': int(df['age_18_greater'].fillna(0).sum())
        }
        
        suggestions.append({
            'field': 'age_distribution',
            'title': 'Enrollment by Age Group',
            'recommended_chart': 'pie',
            'data': [{'name': k, 'value': v} for k, v in age_totals.items()],
            'insight': f"Dominant age group: 0-5 years with {age_totals['0-5 Years']:,} enrollments"
        })
    
    # Chart 3: Monthly trend (if date available)
    if 'date' in df.columns:
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        df_copy['month'] = df_copy['date'].dt.to_period('M').astype(str)
        monthly = df_copy.groupby('month').size().reset_index(name='count')
        monthly = monthly.sort_values('month').tail(10)
        
        suggestions.append({
            'field': 'date',
            'title': 'Monthly Enrollment Trend',
            'recommended_chart': 'line',
            'data': [{'name': r['month'], 'value': int(r['count'])} for _, r in monthly.iterrows()],
            'insight': 'Shows enrollment activity over time'
        })
    
    # Chart 4: Intervention areas from ML (if available)
    if ml_results and 'intervention_areas' in ml_results:
        intervention = ml_results['intervention_areas'][:8]
        suggestions.append({
            'field': 'intervention',
            'title': 'Districts Requiring Intervention (Severity Score)',
            'recommended_chart': 'bar',
            'data': [{'name': a['district'][:15], 'value': round(a['severity_score'], 1)} for a in intervention],
            'insight': f"Critical: {intervention[0]['district']} needs immediate attention" if intervention else ''
        })
    
    return suggestions


def _generate_field_suggestion(df, field, ml_results):
    """Generate suggestion for a specific field"""
    
    if field in ['age_0_5', 'age_5_17', 'age_18_greater']:
        # State-wise breakdown for age group
        if 'state' in df.columns:
            state_breakdown = df.groupby('state')[field].sum().sort_values(ascending=False).head(10)
            age_labels = {'age_0_5': '0-5 Years', 'age_5_17': '5-17 Years', 'age_18_greater': '18+ Years'}
            return {
                'field': field,
                'title': f'{age_labels.get(field, field)} Enrollment by State',
                'recommended_chart': 'bar',
                'data': [{'name': str(k), 'value': int(v)} for k, v in state_breakdown.items()],
                'insight': f"Top state for {age_labels.get(field, field)}: {state_breakdown.index[0]}"
            }
    
    elif field == 'state':
        if 'age_0_5' in df.columns:
            state_totals = df.groupby('state').agg({
                'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'
            }).sum(axis=1).sort_values(ascending=False).head(10)
        else:
            state_totals = df['state'].value_counts().head(10)
        
        return {
            'field': 'state',
            'title': 'Top States by Enrollment',
            'recommended_chart': 'bar',
            'data': [{'name': str(k), 'value': int(v)} for k, v in state_totals.items()],
            'insight': f"Leading state: {state_totals.index[0]}"
        }
    
    elif field == 'district':
        district_counts = df['district'].value_counts().head(10)
        return {
            'field': 'district',
            'title': 'Top Districts by Records',
            'recommended_chart': 'bar',
            'data': [{'name': str(k)[:20], 'value': int(v)} for k, v in district_counts.items()],
            'insight': f"Most active: {district_counts.index[0]}"
        }
    
    else:
        # Generic field
        value_counts = df[field].value_counts().head(10)
        return {
            'field': field,
            'title': f'{field} Distribution',
            'recommended_chart': 'bar',
            'data': [{'name': str(k), 'value': int(v)} for k, v in value_counts.items()],
            'insight': f"Top value: {value_counts.index[0]}"
        }
