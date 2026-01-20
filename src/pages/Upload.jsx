import { Upload as UploadIcon, FileText, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import { useState, useRef } from 'react'
import { API_BASE_URL } from "../api";

const Upload = () => {
  const [dragActive, setDragActive] = useState(false)
  const [file, setFile] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef(null)

  const validateFile = (file) => {
    const validTypes = [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/json',
    ]
    const validExtensions = ['.csv', '.xlsx', '.xls', '.json']
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase()

    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
      setError('Please upload a CSV, Excel, or JSON file')
      return false
    }

    if (file.size > 100 * 1024 * 1024) {
      setError('File size must be less than 100MB')
      return false
    }

    setError(null)
    return true
  }

  const handleFiles = (files) => {
    if (files && files.length > 0) {
      const selectedFile = files[0]
      if (validateFile(selectedFile)) {
        setFile(selectedFile)
      }
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (!loading) {
      handleFiles(e.dataTransfer.files)
    }
  }

  const handleChange = (e) => {
    e.preventDefault()
    handleFiles(e.target.files)
  }

  const onButtonClick = () => {
    if (!loading) {
      fileInputRef.current?.click()
    }
  }

  const handleProcessDataset = async () => {
    if (!file) return;
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Upload failed");
      }

      console.log("Upload success:", data);

      // Redirect to schema page
      window.location.href = "/schema";

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-20">
      <div className="mb-12">
        <h1 className="text-4xl mb-4">Upload Dataset</h1>
        <p className="text-text-muted">Supports CSV, JSON, and Excel files up to 100MB.</p>
      </div>

      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
        className={`
          relative border-2 border-dashed rounded-2xl p-20 flex flex-col items-center justify-center transition-all cursor-pointer
          ${dragActive ? 'border-primary bg-primary/5' : 'border-background-tertiary bg-background-secondary hover:border-primary/50'}
          ${file ? 'border-green-500/50 bg-green-500/5' : ''}
          ${error ? 'border-red-500/50 bg-red-500/5' : ''}
          ${loading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".csv,.xlsx,.xls,.json"
          onChange={handleChange}
          disabled={loading}
        />

        <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-6 transition-colors ${file ? 'bg-green-500/20 text-green-500' :
            error ? 'bg-red-500/20 text-red-500' :
              'bg-primary/10 text-primary'
          }`}>
          {loading ? (
            <Loader2 className="w-8 h-8 animate-spin" />
          ) : file ? (
            <CheckCircle2 className="w-8 h-8" />
          ) : error ? (
            <AlertCircle className="w-8 h-8" />
          ) : (
            <UploadIcon className="w-8 h-8" />
          )}
        </div>

        <div className="text-center">
          <h3 className="text-xl mb-2">
            {loading ? 'Uploading...' : file ? file.name : 'Drop your file here'}
          </h3>
          <p className="text-text-muted mb-8">
            {file ? formatFileSize(file.size) : 'or click to browse (CSV, Excel, JSON)'}
          </p>

          <button
            className="px-6 py-2 bg-background-tertiary hover:bg-background-tertiary/80 rounded-lg text-sm font-medium transition-colors"
            onClick={(e) => {
              e.stopPropagation()
              onButtonClick()
            }}
            disabled={loading}
          >
            {file ? 'Change File' : 'Select File'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {file && !error && (
        <div className="mt-12 p-6 rounded-xl bg-background-secondary border border-background-tertiary flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-background-tertiary flex items-center justify-center">
              <FileText className="w-5 h-5 text-text-muted" />
            </div>
            <div>
              <div className="font-medium">Ready for analysis</div>
              <div className="text-xs text-text-dim uppercase tracking-widest">{file.name.split('.').pop().toUpperCase()}</div>
            </div>
          </div>
          <button
            onClick={handleProcessDataset}
            disabled={loading}
            className="px-8 py-3 bg-primary hover:bg-primary-dark text-white rounded-lg font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            {loading ? 'Processing...' : 'Process Dataset'}
          </button>
        </div>
      )}
    </div>
  )
}

export default Upload
