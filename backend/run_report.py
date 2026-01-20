import sys
import os

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

from app.report import generate_report
from app.ml_service import ml_service

print("Loading ML results...")
results = ml_service.get_results()

if results:
    print("Generating report...")
    report_path = generate_report({'ml_insights': results})
    print(f"\n✅ Report saved to: {report_path}")
    print(f"📁 Full path: {os.path.abspath(report_path)}")
else:
    print("❌ No ML results found. Please run ML analysis first.")
