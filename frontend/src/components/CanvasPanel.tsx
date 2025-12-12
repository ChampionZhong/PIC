import { useState, useEffect, useRef, useLayoutEffect } from 'react';
import mermaid from 'mermaid';
import { ZoomIn, ZoomOut, RotateCcw, Sparkles, Download, RefreshCw, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';

interface CanvasPanelProps {
  code: string;
}

interface RenderResult {
  schema: string;
  imageUrl: string;
}

// Cache key prefix for localStorage
const RENDER_CACHE_PREFIX = 'render_cache_';

// Simple hash function for mermaid code
const hashCode = (str: string): string => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash).toString(36);
};

// Get cached render result for a mermaid code
const getCachedRender = (code: string): RenderResult | null => {
  if (!code.trim()) return null;
  try {
    const cacheKey = RENDER_CACHE_PREFIX + hashCode(code);
    const cached = localStorage.getItem(cacheKey);
    if (cached) {
      const parsed = JSON.parse(cached);
      // Check if cache is still valid (not expired, e.g., 24 hours)
      const cacheAge = Date.now() - (parsed.timestamp || 0);
      const maxAge = 24 * 60 * 60 * 1000; // 24 hours
      if (cacheAge < maxAge) {
        return parsed.result;
      } else {
        // Remove expired cache
        localStorage.removeItem(cacheKey);
      }
    }
  } catch (err) {
    console.warn('Failed to read render cache:', err);
  }
  return null;
};

// Save render result to cache
const saveCachedRender = (code: string, result: RenderResult): void => {
  if (!code.trim()) return;
  try {
    const cacheKey = RENDER_CACHE_PREFIX + hashCode(code);
    const cacheData = {
      result,
      timestamp: Date.now(),
    };
    localStorage.setItem(cacheKey, JSON.stringify(cacheData));
  } catch (err) {
    console.warn('Failed to save render cache:', err);
    // If storage is full, try to clear old entries
    try {
      const keys = Object.keys(localStorage);
      const cacheKeys = keys.filter(k => k.startsWith(RENDER_CACHE_PREFIX));
      // Remove oldest 50% of cache entries
      const entries = cacheKeys.map(key => ({
        key,
        timestamp: JSON.parse(localStorage.getItem(key) || '{}').timestamp || 0,
      })).sort((a, b) => a.timestamp - b.timestamp);
      entries.slice(0, Math.floor(entries.length / 2)).forEach(entry => {
        localStorage.removeItem(entry.key);
      });
      // Retry saving
      localStorage.setItem(RENDER_CACHE_PREFIX + hashCode(code), JSON.stringify({
        result,
        timestamp: Date.now(),
      }));
    } catch (retryErr) {
      console.warn('Failed to clear cache and retry:', retryErr);
    }
  }
};

type RenderStep = 'idle' | 'analyzing' | 'designing' | 'rendering' | 'complete';

/**
 * Simple cleanup: only fix the most common issues that LLM might generate
 */
const cleanMermaidCode = (code: string): string => {
  let cleaned = code.trim();
  cleaned = cleaned.replace(/<[^>]+>/g, ' ');
  cleaned = cleaned.replace(/\[([^\]]*)\]/g, (_match, content) => {
    return '[' + content.replace(/\n/g, ' ').replace(/\s+/g, ' ') + ']';
  });
  cleaned = cleaned.replace(/\|([^|]*)\|/g, (_match, content) => {
    return '|' + content.replace(/\n/g, ' ').replace(/\s+/g, ' ') + '|';
  });
  cleaned = cleaned.replace(/[ \t]+/g, ' ');
  return cleaned.trim();
};

const CanvasPanel: React.FC<CanvasPanelProps> = ({ code }) => {
  const [activeTab, setActiveTab] = useState<'blueprint' | 'rendering'>('blueprint');
  const [renderStep, setRenderStep] = useState<RenderStep>('idle');
  const [renderResult, setRenderResult] = useState<RenderResult | null>(null);
  const [schemaExpanded, setSchemaExpanded] = useState(false);
  
  const mermaidRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const scaledWrapperRef = useRef<HTMLDivElement>(null);
  const [mermaidError, setMermaidError] = useState<string | null>(null); // Separate error for Mermaid
  const [renderError, setRenderError] = useState<string | null>(null); // Separate error for rendering
  
  // Load cached render result when code changes
  useEffect(() => {
    if (code.trim()) {
      const cached = getCachedRender(code);
      if (cached) {
        console.log('Loading cached render result for code');
        setRenderResult(cached);
        setRenderStep('complete');
        setRenderError(null);
      } else {
        // If no cache and we have a result, clear it (code changed)
        if (renderResult) {
          setRenderResult(null);
          setRenderStep('idle');
        }
      }
    } else {
      setRenderResult(null);
      setRenderStep('idle');
    }
  }, [code]); // eslint-disable-line react-hooks/exhaustive-deps
  const [isInitialized, setIsInitialized] = useState(false);
  const [scale, setScale] = useState(1);
  const [baseScale, setBaseScale] = useState(1); // Base scale to fit container at 100%
  const MIN_SCALE = 0.25;
  const MAX_SCALE = 8;
  const SCALE_STEP = 0.25;

  // Initialize mermaid once on mount
  useEffect(() => {
    if (!isInitialized) {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'loose',
        flowchart: {
          useMaxWidth: true,
          htmlLabels: true,
          curve: 'basis',
        },
      });
      setIsInitialized(true);
    }
  }, [isInitialized]);

  // Render diagram when code changes (only in blueprint tab)
  useEffect(() => {
    if (activeTab !== 'blueprint' || !isInitialized || !mermaidRef.current) {
      return;
    }

    if (!code || code.trim() === '') {
      mermaidRef.current.innerHTML = '';
      setMermaidError(null);
      return;
    }

    mermaidRef.current.innerHTML = '';
    setMermaidError(null);

    const cleanedCode = cleanMermaidCode(code);
    const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;

    mermaid
      .render(id, cleanedCode)
      .then((result) => {
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = result.svg;
          setMermaidError(null);
          setScale(1);
          
          // Ensure SVG displays correctly and calculate base scale for 100% fit
          const svgElement = mermaidRef.current.querySelector('svg');
          if (svgElement && containerRef.current) {
            // Remove width="100%" if present, as it causes issues with scaling
            if (svgElement.getAttribute('width') === '100%') {
              svgElement.removeAttribute('width');
            }
            
            // Remove max-width from style that prevents scaling
            if (svgElement.hasAttribute('style')) {
              const currentStyle = svgElement.getAttribute('style') || '';
              // Remove max-width from style to allow scaling
              const newStyle = currentStyle.replace(/max-width:\s*[^;]+;?/g, '').trim();
              svgElement.setAttribute('style', newStyle || '');
            }
            
            // Ensure SVG has explicit dimensions from viewBox for proper rendering
            const viewBox = svgElement.getAttribute('viewBox');
            if (viewBox) {
              const [, , vbWidth, vbHeight] = viewBox.split(' ').map(Number);
              
              // Set width/height from viewBox to ensure SVG renders
              if (!svgElement.getAttribute('width')) {
                svgElement.setAttribute('width', vbWidth.toString());
                svgElement.setAttribute('height', vbHeight.toString());
              }
              
              // Calculate base scale to fit container at 100%
              const containerWidth = containerRef.current.clientWidth;
              const containerHeight = containerRef.current.clientHeight;
              const padding = 64; // 2rem * 2
              const availableWidth = containerWidth - padding;
              const availableHeight = containerHeight - padding;
              
              // Calculate scale to fit both width and height, use the smaller one
              const scaleX = availableWidth / vbWidth;
              const scaleY = availableHeight / vbHeight;
              const calculatedBaseScale = Math.min(scaleX, scaleY, 1); // Don't scale up beyond 100%
              
              setBaseScale(calculatedBaseScale);
              setScale(1); // Reset to 100% (which will use baseScale)
            }
          }
        }
      })
      .catch((err) => {
        console.error('Mermaid rendering error:', err);
        const errorMessage = err instanceof Error ? err.message : 'Failed to render diagram';
        setMermaidError(errorMessage);
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = '';
        }
      });
  }, [code, isInitialized, activeTab]);

  // Update scaled wrapper - allow free scaling with overflow handling
  useLayoutEffect(() => {
    if (mermaidRef.current && scaledWrapperRef.current && containerRef.current && activeTab === 'blueprint') {
      const svgElement = mermaidRef.current.querySelector('svg');
      if (svgElement) {
        // Calculate actual scale (baseScale * userScale)
        const actualScale = baseScale * scale;
        
        // Apply scale transform - use top-left origin to prevent scroll issues
        scaledWrapperRef.current.style.transform = `scale(${actualScale})`;
        scaledWrapperRef.current.style.transformOrigin = 'top left';
        
        // Remove any max-width restrictions that prevent scaling
        svgElement.style.maxWidth = '';
        svgElement.style.maxHeight = '';
        
        // Ensure SVG maintains its natural dimensions for proper scaling
        const viewBox = svgElement.getAttribute('viewBox');
        if (viewBox && !svgElement.getAttribute('width')) {
          const [, , vbWidth, vbHeight] = viewBox.split(' ').map(Number);
          svgElement.setAttribute('width', vbWidth.toString());
          svgElement.setAttribute('height', vbHeight.toString());
        }
        
        // Ensure the wrapper has the correct dimensions for scrolling
        // The wrapper should be the size of the scaled SVG
        if (viewBox) {
          const [, , vbWidth, vbHeight] = viewBox.split(' ').map(Number);
          scaledWrapperRef.current.style.width = `${vbWidth * actualScale}px`;
          scaledWrapperRef.current.style.height = `${vbHeight * actualScale}px`;
        }
      }
    }
  }, [scale, baseScale, code, isInitialized, activeTab]);

  const handleZoomIn = () => {
    setScale((prev) => Math.min(prev + SCALE_STEP, MAX_SCALE));
  };

  const handleZoomOut = () => {
    setScale((prev) => Math.max(prev - SCALE_STEP, MIN_SCALE));
  };

  const handleResetZoom = () => {
    setScale(1);
  };

  const handleGenerate = async () => {
    if (!code.trim()) {
      return;
    }

    // Switch to rendering tab
    setActiveTab('rendering');
    setRenderStep('analyzing');
    setRenderResult(null);
    setRenderError(null);

    try {
      // Step 1: Analyzing Topology
      await new Promise((resolve) => setTimeout(resolve, 800));

      // Step 2: Designing Visual Schema
      setRenderStep('designing');
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Step 3: Rendering Pixels
      setRenderStep('rendering');

      // Call the API
      const response = await fetch('http://localhost:8000/api/render', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mermaid_code: code,
          style: 'neurips',
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: RenderResult = await response.json();
      console.log('Render result:', data); // Debug log
      // Handle both camelCase and snake_case field names
      const normalizedData: RenderResult = {
        schema: data.schema || (data as any).visual_schema || '',
        imageUrl: data.imageUrl || (data as any).image_url || '',
      };
      if (!normalizedData.imageUrl) {
        throw new Error('No image URL in response');
      }
      setRenderResult(normalizedData);
      setRenderStep('complete');
      setRenderError(null);
      // Save to cache
      saveCachedRender(code, normalizedData);
    } catch (err) {
      console.error('Render error:', err);
      setRenderError(err instanceof Error ? err.message : 'Failed to generate image');
      setRenderStep('idle');
    }
  };

  const handleDownload = async () => {
    if (!renderResult?.imageUrl) {
      setRenderError('No image available to download');
      return;
    }

    try {
      // Handle both data URLs and regular URLs
      if (renderResult.imageUrl.startsWith('data:')) {
        // Data URL - create download link
        const link = document.createElement('a');
        link.href = renderResult.imageUrl;
        link.download = `research-figure-${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } else {
        // Regular URL - fetch and download
        // Add CORS mode for cross-origin requests
        const response = await fetch(renderResult.imageUrl, {
          mode: 'cors',
          credentials: 'omit',
        });
        
        if (!response.ok) {
          throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
        }
        
        const blob = await response.blob();
        if (!blob || blob.size === 0) {
          throw new Error('Downloaded image is empty');
        }
        
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `research-figure-${Date.now()}.png`;
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        
        // Clean up after a delay
        setTimeout(() => {
          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);
        }, 100);
      }
    } catch (err) {
      console.error('Download error:', err);
      setRenderError(err instanceof Error ? err.message : 'Failed to download image');
    }
  };

  const handleRegenerate = () => {
    handleGenerate();
  };

  const renderProgressSteps = () => {
    const steps = [
      { key: 'analyzing', label: 'Analyzing Topology...', icon: Loader2 },
      { key: 'designing', label: 'Designing Visual Schema...', icon: Loader2 },
      { key: 'rendering', label: 'Rendering Pixels...', icon: Loader2 },
    ];

    const getStepStatus = (stepIndex: number): 'idle' | 'active' | 'complete' => {
      if (renderStep === 'complete') return 'complete';
      const currentStepIndex = steps.findIndex((s) => s.key === renderStep);
      if (currentStepIndex === -1) return 'idle';
      if (stepIndex < currentStepIndex) return 'complete';
      if (stepIndex === currentStepIndex) return 'active';
      return 'idle';
    };

    return (
      <div className="space-y-4 py-8">
        {steps.map((step, index) => {
          const status = getStepStatus(index);
          const isActive = status === 'active';
          const isComplete = status === 'complete';
          const Icon = step.icon;

          return (
            <div
              key={step.key}
              className={`flex items-center gap-4 px-6 py-4 rounded-xl border transition-all ${
                isActive
                  ? 'bg-blue-50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800 shadow-sm'
                  : isComplete
                  ? 'bg-green-50 dark:bg-green-950/20 border-green-200 dark:border-green-800'
                  : 'bg-zinc-50 dark:bg-slate-800/50 border-zinc-200 dark:border-slate-700'
              }`}
            >
              <div
                className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${
                  isActive
                    ? 'bg-blue-500 text-white animate-spin'
                    : isComplete
                    ? 'bg-green-500 text-white'
                    : 'bg-zinc-200 dark:bg-slate-700 text-zinc-500 dark:text-slate-400'
                }`}
              >
                {isComplete ? (
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  <Icon className={`w-5 h-5 ${isActive ? 'animate-spin' : ''}`} />
                )}
              </div>
              <div className="flex-1">
                <div
                  className={`font-medium ${
                    isActive
                      ? 'text-blue-900 dark:text-blue-100'
                      : isComplete
                      ? 'text-green-900 dark:text-green-100'
                      : 'text-zinc-600 dark:text-slate-400'
                  }`}
                >
                  {step.label}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="w-full h-full bg-zinc-50 dark:bg-slate-900 flex flex-col">
      {/* Tab Selector */}
      <div className="flex items-center justify-between gap-4 px-4 pt-4 pb-3 border-b border-zinc-200 dark:border-slate-800">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab('blueprint')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all ${
              activeTab === 'blueprint'
                ? 'bg-zinc-900 dark:bg-slate-800 text-white shadow-sm'
                : 'bg-white dark:bg-slate-800/50 text-zinc-600 dark:text-slate-400 hover:bg-zinc-100 dark:hover:bg-slate-800'
            }`}
          >
            <span>🛠️</span>
            <span>Blueprint</span>
          </button>
          <button
            onClick={() => setActiveTab('rendering')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all ${
              activeTab === 'rendering'
                ? 'bg-zinc-900 dark:bg-slate-800 text-white shadow-sm'
                : 'bg-white dark:bg-slate-800/50 text-zinc-600 dark:text-slate-400 hover:bg-zinc-100 dark:hover:bg-slate-800'
            }`}
          >
            <span>🖼️</span>
            <span>Rendering</span>
          </button>
        </div>
        {/* Render Button */}
        {code.trim() && (
          <button
            onClick={handleGenerate}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium rounded-lg shadow-sm hover:shadow-md transition-all text-sm"
          >
            <Sparkles className="w-4 h-4" />
            <span>Render High-Fidelity Figure</span>
          </button>
        )}
      </div>

      {/* Content Area */}
      <div className="flex-1 min-h-0 relative">
        {activeTab === 'blueprint' ? (
          <>
            {/* Zoom Controls */}
            {!mermaidError && code.trim() && (
              <div className="absolute bottom-4 right-4 z-10 flex items-center gap-2 bg-white dark:bg-slate-800/90 backdrop-blur-sm border border-zinc-200 dark:border-slate-700 rounded-xl p-1.5 shadow-sm">
                <button
                  onClick={handleZoomOut}
                  disabled={scale <= MIN_SCALE}
                  className="p-1.5 text-zinc-600 dark:text-slate-300 hover:text-zinc-900 dark:hover:text-slate-100 hover:bg-zinc-100 dark:hover:bg-slate-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Zoom out"
                >
                  <ZoomOut className="w-4 h-4" />
                </button>
                <span className="px-2 text-xs text-zinc-500 dark:text-slate-400 min-w-[3rem] text-center">
                  {Math.round(scale * 100)}%
                </span>
                <button
                  onClick={handleZoomIn}
                  disabled={scale >= MAX_SCALE}
                  className="p-1.5 text-zinc-600 dark:text-slate-300 hover:text-zinc-900 dark:hover:text-slate-100 hover:bg-zinc-100 dark:hover:bg-slate-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Zoom in"
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
                <div className="w-px h-4 bg-zinc-200 dark:bg-slate-700 mx-1" />
                <button
                  onClick={handleResetZoom}
                  disabled={scale === 1}
                  className="p-1.5 text-zinc-600 dark:text-slate-300 hover:text-zinc-900 dark:hover:text-slate-100 hover:bg-zinc-100 dark:hover:bg-slate-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Reset zoom"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Diagram Container */}
            <div ref={containerRef} className="flex-1 overflow-auto bg-white dark:bg-slate-900 h-full">
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  justifyContent: 'flex-start',
                  minHeight: '100%',
                  width: '100%',
                  padding: '2rem',
                  boxSizing: 'border-box',
                }}
              >
                <div
                  ref={scaledWrapperRef}
                  style={{
                    display: 'inline-block',
                  }}
                >
                  {mermaidError ? (
                    <div className="text-red-600 dark:text-red-400 p-4 text-sm bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/50 rounded-xl max-w-2xl">
                      <div className="font-semibold mb-2">Error rendering diagram:</div>
                      <div className="text-red-500 dark:text-red-300">{mermaidError}</div>
                    </div>
                  ) : code.trim() ? (
                    <div
                      ref={mermaidRef}
                      className="mermaid-container"
                      style={{
                        display: 'inline-block',
                        maxWidth: '100%',
                      }}
                    />
                  ) : (
                    <div className="text-center text-zinc-500 dark:text-slate-400">
                      <div className="text-4xl mb-4">📊</div>
                      <p className="text-lg font-medium text-zinc-700 dark:text-slate-300">Live Blueprint Preview</p>
                      <p className="text-sm mt-2 text-zinc-500 dark:text-slate-500">
                        Start chatting to generate a diagram
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

          </>
        ) : (
          <div className="h-full overflow-auto bg-white dark:bg-slate-900 p-6">
            {renderStep === 'idle' && !renderResult && (
              <div className="h-full flex items-center justify-center">
                <div className="text-center text-zinc-500 dark:text-slate-400">
                  <div className="text-5xl mb-4">🎨</div>
                  <p className="text-xl font-medium text-zinc-700 dark:text-slate-300 mb-2">
                    High-Fidelity Rendering
                  </p>
                  <p className="text-sm">
                    Switch to Blueprint tab and click "Render High-Fidelity Figure" to generate
                  </p>
                </div>
              </div>
            )}

            {(renderStep === 'analyzing' || renderStep === 'designing' || renderStep === 'rendering') && (
              <div className="max-w-2xl mx-auto">
                <div className="text-center mb-8">
                  <h2 className="text-2xl font-semibold text-zinc-900 dark:text-slate-100 mb-2">
                    Generating Your Figure
                  </h2>
                  <p className="text-sm text-zinc-600 dark:text-slate-400">
                    This may take a moment...
                  </p>
                </div>
                {renderProgressSteps()}
              </div>
            )}

            {renderStep === 'complete' && renderResult && (
              <div className="max-w-4xl mx-auto space-y-6">
                {/* Image Display */}
                <div className="bg-white dark:bg-slate-800 rounded-xl border border-zinc-200 dark:border-slate-700 shadow-sm overflow-hidden">
                  <div className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-zinc-900 dark:text-slate-100">
                        Generated Figure
                      </h3>
                      <div className="flex gap-2">
                        <button
                          onClick={handleDownload}
                          className="flex items-center gap-2 px-4 py-2 bg-zinc-100 dark:bg-slate-700 hover:bg-zinc-200 dark:hover:bg-slate-600 text-zinc-900 dark:text-slate-100 rounded-lg text-sm font-medium transition-colors"
                        >
                          <Download className="w-4 h-4" />
                          Download
                        </button>
                        <button
                          onClick={handleRegenerate}
                          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                        >
                          <RefreshCw className="w-4 h-4" />
                          Regenerate
                        </button>
                      </div>
                    </div>
                    <div className="rounded-lg overflow-hidden border border-zinc-200 dark:border-slate-700 bg-zinc-50 dark:bg-slate-900">
                      <img
                        src={renderResult.imageUrl}
                        alt="Generated research figure"
                        className="w-full h-auto"
                        onError={(e) => {
                          console.error('Image load error:', e, 'Image URL:', renderResult.imageUrl);
                          setRenderError(`Failed to load image. URL: ${renderResult.imageUrl.substring(0, 100)}...`);
                        }}
                        onLoad={() => {
                          console.log('Image loaded successfully:', renderResult.imageUrl);
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Visual Schema Accordion */}
                <div className="bg-white dark:bg-slate-800 rounded-xl border border-zinc-200 dark:border-slate-700 shadow-sm overflow-hidden">
                  <button
                    onClick={() => setSchemaExpanded(!schemaExpanded)}
                    className="w-full px-6 py-4 flex items-center justify-between hover:bg-zinc-50 dark:hover:bg-slate-700/50 transition-colors"
                  >
                    <span className="font-medium text-zinc-900 dark:text-slate-100">
                      Visual Schema (Debug)
                    </span>
                    {schemaExpanded ? (
                      <ChevronUp className="w-5 h-5 text-zinc-500 dark:text-slate-400" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-zinc-500 dark:text-slate-400" />
                    )}
                  </button>
                  {schemaExpanded && (
                    <div className="px-6 pb-4 border-t border-zinc-200 dark:border-slate-700">
                      <pre className="mt-4 p-4 bg-zinc-50 dark:bg-slate-900 rounded-lg text-xs font-mono text-zinc-800 dark:text-slate-300 overflow-auto max-h-96">
                        {renderResult.schema}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            )}

            {renderError && renderStep === 'idle' && (
              <div className="max-w-2xl mx-auto">
                <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/50 rounded-xl p-6">
                  <div className="font-semibold text-red-900 dark:text-red-100 mb-2">Error</div>
                  <div className="text-red-700 dark:text-red-300 text-sm">{renderError}</div>
                  <button
                    onClick={handleRegenerate}
                    className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CanvasPanel;

