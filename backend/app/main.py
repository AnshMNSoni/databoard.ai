from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
import os
import shutil
import glob

from app.config import UPLOAD_DIR
from app.data_store import data_store
from app.utils import load_dataset, detect_schema
from app.analytics import analyze_data
from app.ai_engine import suggest_charts
from app.report import generate_report
from app.schemas import AnalyzeRequest, SuggestRequest
from app.ml_service import ml_service

app = FastAPI(title="AI Analytics Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Auto-load last CSV file on startup so schema is available
def _load_last_csv():
    csv_files = glob.glob(os.path.join(UPLOAD_DIR, "*.csv"))
    if csv_files:
        # Load the most recently modified file
        latest_file = max(csv_files, key=os.path.getmtime)
        try:
            df = load_dataset(latest_file)
            data_store.df = df
            data_store.filename = os.path.basename(latest_file)
            print(f"✓ Auto-loaded dataset: {data_store.filename} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠ Could not auto-load CSV: {e}")

_load_last_csv()

@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = load_dataset(file_path)

    data_store.df = df
    data_store.filename = file.filename
    
    # Clear old ML results and start fresh analysis
    ml_service.clear_cache(file.filename)
    background_tasks.add_task(ml_service.run_analysis, [file_path])
    
    return {
        "message": "File uploaded successfully",
        "rows": len(df),
        "filename": file.filename,
        "ml_status": "started"
    }

@app.get("/api/schema")
def get_schema():
    df = data_store.df
    if df is None:
        return {"error": "No dataset uploaded"}

    schema = detect_schema(df)
    return {"schema": schema}

@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    df = data_store.df
    if df is None:
        return {"error": "No dataset uploaded"}

    result = analyze_data(df, request.fields)
    
    # Get ML insights to enhance analysis
    ml_results = ml_service.get_results()
    
    # Build AI insights for Gemini to use
    ai_insights = []
    if ml_results:
        if 'auto_generated_insights' in ml_results:
            ai_insights = ml_results['auto_generated_insights'][:10]
        
        # Add model prediction insights
        if 'model_comparison' in ml_results:
            best_model = min(ml_results['model_comparison'].items(), 
                           key=lambda x: x[1].get('mae', float('inf')))
            ai_insights.append(f"Best forecasting model: {best_model[0]} with MAE of {best_model[1].get('mae', 0):.2f}")
        
        # Add intervention insights
        if 'intervention_areas' in ml_results and ml_results['intervention_areas']:
            top_intervention = ml_results['intervention_areas'][0]
            ai_insights.append(f"Priority intervention: {top_intervention['district']} (severity: {top_intervention['severity_score']:.1f})")
    
    return {
        "analysis": result,
        "ai_insights": ai_insights,
        "ml_available": ml_results is not None
    }

@app.post("/api/suggest")
def suggest(request: SuggestRequest):
    suggestions = suggest_charts(request.fields)
    
    # Enhance suggestions with ML insights
    ml_results = ml_service.get_results()
    ai_recommendations = []
    
    if ml_results:
        if 'auto_generated_insights' in ml_results:
            ai_recommendations = ml_results['auto_generated_insights'][:5]
    
    return {
        "suggestions": suggestions,
        "ai_recommendations": ai_recommendations
    }

@app.post("/api/dashboard")
def generate_dashboard(request: AnalyzeRequest):
    df = data_store.df
    if df is None:
        return {"error": "No dataset uploaded"}

    analysis = analyze_data(df, request.fields)
    charts = suggest_charts(request.fields)
    
    # Get ML results for dashboard insights
    ml_results = ml_service.get_results()
    
    # Build comprehensive dashboard with AI insights
    ai_insights = []
    key_findings = []
    
    if ml_results:
        # Auto-generated insights
        if 'auto_generated_insights' in ml_results:
            ai_insights = ml_results['auto_generated_insights']
        
        # Key findings from model comparison
        if 'model_comparison' in ml_results:
            key_findings.append({
                'type': 'models',
                'title': 'ML Model Performance',
                'data': ml_results['model_comparison']
            })
        
        # State analysis findings
        if 'state_analysis' in ml_results:
            key_findings.append({
                'type': 'states',
                'title': 'State Analysis',
                'top_states': ml_results['state_analysis'][:5] if isinstance(ml_results['state_analysis'], list) else []
            })
        
        # Intervention areas
        if 'intervention_areas' in ml_results:
            key_findings.append({
                'type': 'intervention',
                'title': 'Priority Intervention Areas',
                'areas': ml_results['intervention_areas'][:10]
            })
        
        # Forecasts
        if 'forecasts' in ml_results:
            key_findings.append({
                'type': 'forecast',
                'title': 'Enrollment Forecasts',
                'data': ml_results['forecasts']
            })

    dashboard = {
        "dataset": data_store.filename,
        "analysis": {**analysis, "auto_generated_insights": ai_insights},
        "charts": charts,
        "key_findings": key_findings,
        "ml_status": "completed" if ml_results else "pending"
    }

    return dashboard

@app.get("/api/report")
def download_report():
    df = data_store.df
    if df is None:
        return {"error": "No dataset uploaded"}

    analysis = analyze_data(df, df.columns.tolist())

    ml_results = ml_service.get_results()

    if ml_results is None:
        ml_results = {
            "auto_generated_insights": []
        }

    full_report = {
        "analysis": analysis,
        "ml_insights": ml_results
    }

    print("ML RESULTS FOR REPORT:", ml_results)  # Debug

    report_path = generate_report(full_report)

    return FileResponse(report_path, filename="analytics_report.pdf")

@app.post("/api/ml/run")
def run_ml_analysis(background_tasks: BackgroundTasks):
    files = glob.glob("uploads/*.csv")

    if not files:
        return {"error": "No CSV files found for ML analysis"}

    background_tasks.add_task(ml_service.run_analysis, files)

    return {
        "message": "ML analysis started",
        "status": "processing",
        "files": files
    }

@app.get("/api/ml/results")
def get_ml_results():
    status = ml_service.get_status()
    results = ml_service.get_results()

    if status["is_running"]:
        return {"status": "processing", "message": "ML analysis in progress..."}
    
    if not results:
        return {"status": "no_data", "message": "No ML results available. Upload a file first."}

    return {
        "status": "completed",
        "results": results
    }

@app.get("/api/ml/status")
def get_ml_status():
    """Check if ML analysis is running or complete"""
    status = ml_service.get_status()
    return status

@app.get("/api/ping")
def ping():
    return {"status": "ok", "message": "Backend is running"}
