import { FileText } from "lucide-react";
import { API_BASE_URL } from "../api";

const Report = () => {
  const downloadReport = () => {
    window.open(`${API_BASE_URL}/api/report`, "_blank");
  };

  return (
    <div className="max-w-4xl mx-auto px-6 py-20">
      <h1 className="text-4xl mb-6">Analytics Report</h1>

      <button
        onClick={downloadReport}
        className="px-8 py-3 bg-primary text-white rounded-lg flex items-center gap-2"
      >
        <FileText className="w-5 h-5" />
        Download Report
      </button>
    </div>
  );
};

export default Report;
