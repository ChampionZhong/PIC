import { useState, useRef } from 'react';

const API_BASE_URL = 'http://localhost:8000';

interface FigureToPPTProps {
  // Add any props if needed
}

export default function FigureToPPT({}: FigureToPPTProps) {
  const [draftImageFile, setDraftImageFile] = useState<File | null>(null);
  const [draftImagePreview, setDraftImagePreview] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [maskDetailLevel, setMaskDetailLevel] = useState<number>(3);
  const [figureComplex, setFigureComplex] = useState<string>('medium');
  const [mineruPort, setMineruPort] = useState<number | undefined>(undefined);
  
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [pptFile, setPptFile] = useState<Blob | null>(null);
  
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setDraftImageFile(file);
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setDraftImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      setError('');
    }
  };
  
  const handleGenerate = async () => {
    if (!draftImageFile) {
      setError('Please select a draft image file');
      return;
    }
    
    setIsProcessing(true);
    setStatus('Processing...');
    setError('');
    setPptFile(null);
    
    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('fig_draft_file', draftImageFile);
      formData.append('mask_detail_level', maskDetailLevel.toString());
      formData.append('figure_complex', figureComplex);
      if (mineruPort !== undefined) {
        formData.append('mineru_port', mineruPort.toString());
      }
      
      const response = await fetch(`${API_BASE_URL}/api/figure-to-ppt`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        // Try to parse error as JSON, fallback to text
        let errorMessage = `HTTP ${response.status}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
          const errorText = await response.text();
          errorMessage = errorText || errorMessage;
        }
        throw new Error(errorMessage);
      }
      
      // Check if response is a file (PPT)
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/vnd.openxmlformats')) {
        // Get PPT file as blob
        const blob = await response.blob();
        setPptFile(blob);
        setStatus('PPT generated successfully!');
      } else {
        // Fallback: try to parse as JSON
        const data = await response.json();
        if (data.status === 'success') {
          setStatus('PPT generated successfully!');
        } else {
          throw new Error(data.message || 'Unknown error');
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to process request';
      setError(errorMessage);
      setStatus('Error occurred');
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handleSave = () => {
    if (!pptFile) return;
    
    // Create a download link
    const url = window.URL.createObjectURL(pptFile);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'output.pptx';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };
  
  return (
    <div className="h-full w-full flex flex-col p-6 bg-zinc-900 text-zinc-100 overflow-y-auto">
      <div className="max-w-4xl mx-auto w-full space-y-6">
        <h1 className="text-3xl font-bold mb-6">Figure to PPT Converter</h1>
        
        {/* Input Section */}
        <div className="bg-zinc-800 rounded-lg p-6 space-y-4">
          <h2 className="text-xl font-semibold mb-4">Input Image</h2>
          
          <div>
            <label className="block text-sm font-medium mb-2">
              Draft Image (with content) *
            </label>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
              disabled={isProcessing}
            />
            <div className="flex items-center gap-4">
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessing}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-600 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors"
              >
                Choose File
              </button>
              {draftImageFile && (
                <span className="text-sm text-zinc-300">
                  {draftImageFile.name}
                </span>
              )}
            </div>
            {draftImagePreview && (
              <div className="mt-4">
                <img
                  src={draftImagePreview}
                  alt="Draft preview"
                  className="max-w-full max-h-64 rounded-md border border-zinc-600"
                />
              </div>
            )}
            <p className="text-xs text-zinc-400 mt-2">
              Layout template will be automatically generated from the draft image
            </p>
          </div>
        </div>
        
        {/* Parameters Section */}
        <div className="bg-zinc-800 rounded-lg p-6 space-y-4">
          <h2 className="text-xl font-semibold mb-4">Parameters</h2>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Mask Detail Level
              </label>
              <input
                type="number"
                value={maskDetailLevel}
                onChange={(e) => setMaskDetailLevel(parseInt(e.target.value) || 3)}
                min="1"
                max="10"
                className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isProcessing}
              />
              <p className="text-xs text-zinc-400 mt-1">MinerU recursion depth (1-10)</p>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                Figure Complexity
              </label>
              <select
                value={figureComplex}
                onChange={(e) => setFigureComplex(e.target.value)}
                className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isProcessing}
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                MinerU Port (optional)
              </label>
              <input
                type="number"
                value={mineruPort || ''}
                onChange={(e) => setMineruPort(e.target.value ? parseInt(e.target.value) : undefined)}
                placeholder="Only for local service mode"
                className="w-full px-4 py-2 bg-zinc-700 border border-zinc-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isProcessing}
              />
              <p className="text-xs text-zinc-400 mt-1">Leave empty if using MinerU API (configured via MINERU_API_KEY/URL)</p>
            </div>
          </div>
        </div>
        
        {/* Action Section */}
        <div className="bg-zinc-800 rounded-lg p-6 space-y-3">
          <button
            onClick={handleGenerate}
            disabled={isProcessing || !draftImageFile}
            className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-600 disabled:cursor-not-allowed rounded-md font-medium transition-colors"
          >
            {isProcessing ? 'Processing...' : 'Generate PPT'}
          </button>
          
          {pptFile && (
            <button
              onClick={handleSave}
              className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 rounded-md font-medium transition-colors"
            >
              Save PPT (output.pptx)
            </button>
          )}
        </div>
        
        {/* Status Section */}
        {(status || error || pptFile) && (
          <div className={`rounded-lg p-6 ${
            error ? 'bg-red-900/30 border border-red-700' :
            pptFile ? 'bg-green-900/30 border border-green-700' :
            'bg-zinc-800 border border-zinc-700'
          }`}>
            {error && (
              <div className="text-red-400">
                <h3 className="font-semibold mb-2">Error</h3>
                <p>{error}</p>
              </div>
            )}
            {status && !error && (
              <div className="text-zinc-100">
                <h3 className="font-semibold mb-2">Status</h3>
                <p>{status}</p>
              </div>
            )}
            {pptFile && !error && (
              <div className="mt-4">
                <p className="text-green-400 mb-2">PPT generated successfully!</p>
                <p className="text-sm text-zinc-400">Click "Save PPT" button to download</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

