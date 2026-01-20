import os
import json
from app.aadhaar_model import AadhaarMLModel

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "ml_results.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

class MLService:
    def __init__(self):
        self.model = AadhaarMLModel()
        self.last_results = None
        self.is_running = False
        self.current_file = None
        # Try to load existing results from file on startup
        self._load_cached_results()

    def _load_cached_results(self):
        """Load results from file if available (for when backend restarts)"""
        if os.path.exists(RESULTS_FILE):
            try:
                with open(RESULTS_FILE, 'r') as f:
                    self.last_results = json.load(f)
                print(f"✓ Loaded cached ML results from {RESULTS_FILE}")
            except Exception as e:
                print(f"⚠ Could not load cached results: {e}")
                self.last_results = None

    def clear_cache(self, new_file=None):
        """Clear cached results when new data is uploaded"""
        self.last_results = None
        self.current_file = new_file
        self.is_running = True
        print(f"🔄 Cache cleared - will run fresh ML analysis on: {new_file}")

    def run_analysis(self, csv_files):
        self.is_running = True
        print(f"🚀 Starting ML analysis on: {csv_files}")
        
        try:
            # Load data
            self.model.load_data(csv_files)

            # Run full pipeline
            self.model.run_complete_analysis()

            # Get results
            results = self.model.get_all_results()

            # Save results
            self.model.save_results(RESULTS_FILE)

            self.last_results = results
            self.is_running = False
            print(f"✅ ML analysis completed!")
            return results
        except Exception as e:
            self.is_running = False
            print(f"❌ ML analysis failed: {e}")
            raise e

    def get_status(self):
        return {
            "is_running": self.is_running,
            "has_results": self.last_results is not None,
            "current_file": self.current_file
        }

    def get_results(self):
        # If no results in memory, try loading from file
        if self.last_results is None:
            self._load_cached_results()
        return self.last_results


ml_service = MLService()
