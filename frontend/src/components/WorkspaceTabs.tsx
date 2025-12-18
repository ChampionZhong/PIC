import { useState } from 'react';
import { Code2, Palette, Image as ImageIcon, CheckCircle2, XCircle, Loader2, Download, Sparkles, FileText } from 'lucide-react';
import MermaidRenderer from './MermaidRenderer';
import FigureToPPT from './FigureToPPT';

export type ActiveTab = 'logic' | 'style' | 'result' | 'figure-to-ppt';

interface CriticData {
  feedback: string;
  passed: boolean;
  suggestions?: string;
}

interface WorkspaceTabsProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  mermaidCode: string;
  onMermaidCodeChange: (code: string) => void;
  visualSchema: string;
  onVisualSchemaChange: (schema: string) => void;
  imageUrl: string;
  critique: string;
  onConfirmAndGenerateSchema: () => void;
  onStartRendering: () => void;
  onRegenerate: () => void;
  isGeneratingSchema?: boolean;
  isRendering?: boolean;
  selectedStyle: string;
  onStyleChange: (style: string) => void;
  customStyleText: string;
  onCustomStyleTextChange: (text: string) => void;
  onDownloadImage?: () => void;
  logicCritique?: CriticData | null;
  styleCritique?: CriticData | null;
  resultCritique?: CriticData | null;
  onLogicCritic?: () => void;
  onStyleCritic?: () => void;
  onResultCritic?: () => void;
  isGeneratingLogicCritic?: boolean;
  isGeneratingStyleCritic?: boolean;
  isGeneratingResultCritic?: boolean;
}

const WorkspaceTabs: React.FC<WorkspaceTabsProps> = ({
  activeTab,
  onTabChange,
  mermaidCode,
  onMermaidCodeChange,
  visualSchema,
  onVisualSchemaChange,
  imageUrl,
  critique,
  onConfirmAndGenerateSchema,
  onStartRendering,
  onRegenerate,
  isGeneratingSchema = false,
  isRendering = false,
  selectedStyle,
  onStyleChange,
  customStyleText,
  onCustomStyleTextChange,
  onDownloadImage,
  logicCritique,
  styleCritique,
  resultCritique,
  onLogicCritic,
  onStyleCritic,
  onResultCritic,
  isGeneratingLogicCritic = false,
  isGeneratingStyleCritic = false,
  isGeneratingResultCritic = false,
}) => {
  const tabs: { id: ActiveTab; label: string; icon: React.ReactNode }[] = [
    { id: 'logic', label: 'Logic', icon: <Code2 className="w-4 h-4" /> },
    { id: 'style', label: 'Style', icon: <Palette className="w-4 h-4" /> },
    { id: 'result', label: 'Result', icon: <ImageIcon className="w-4 h-4" /> },
    { id: 'figure-to-ppt', label: 'Figure to PPT', icon: <FileText className="w-4 h-4" /> },
  ];

  // Render Critic Section Component
  const renderCriticSection = (
    critique: CriticData | null | undefined,
    onGenerate: (() => void) | undefined,
    isGenerating: boolean,
    disabledCondition: boolean,
    disabledMessage: string
  ) => {
    return (
      <div className="h-full border-t border-zinc-800 bg-zinc-900/50 flex flex-col">
        <div className="px-4 py-3 flex items-center justify-between border-b border-zinc-800 flex-shrink-0">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <span className="text-sm font-medium text-zinc-300">Critic Review</span>
          </div>
          {onGenerate && (
            <button
              onClick={onGenerate}
              disabled={disabledCondition || isGenerating}
              className="text-xs bg-indigo-600 hover:bg-indigo-700 disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed text-white rounded px-3 py-1.5 font-medium transition-all flex items-center gap-1.5"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="w-3 h-3" />
                  Generate Review
                </>
              )}
            </button>
          )}
        </div>
        {disabledCondition && onGenerate && (
          <div className="px-4 py-2 text-xs text-zinc-500 text-center bg-zinc-900/30 flex-shrink-0">
            {disabledMessage}
          </div>
        )}
        {critique && (
          <div className="flex-1 min-h-0 overflow-y-auto" style={{ maxHeight: '100%' }}>
            <div className="p-4 space-y-3">
              <div className="flex items-start gap-3">
                {critique.passed ? (
                  <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                ) : (
                  <XCircle className="w-5 h-5 text-rose-500 flex-shrink-0 mt-0.5" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-zinc-200 mb-2">
                    {critique.passed ? '✓ Passed' : '✗ Needs Improvement'}
                  </div>
                  <div className="text-sm text-zinc-300 whitespace-pre-wrap leading-relaxed mb-3">
                    {critique.feedback}
                  </div>
                  {critique.suggestions && (
                    <div className="mt-3 pt-3 border-t border-zinc-700">
                      <div className="text-xs font-semibold text-indigo-400 mb-2">💡 Suggestions:</div>
                      <div className="text-sm text-zinc-300 whitespace-pre-wrap leading-relaxed bg-zinc-800/50 rounded p-3">
                        {critique.suggestions}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
        {!critique && !disabledCondition && onGenerate && (
          <div className="px-4 py-6 text-center text-zinc-500 text-sm flex-shrink-0">
            Click "Generate Review" to get feedback and suggestions
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-zinc-950 text-zinc-100">
      {/* Tab Header */}
      <div className="flex border-b border-zinc-800 bg-zinc-900/50">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-colors border-b-2 ${
              activeTab === tab.id
                ? 'border-indigo-500 text-indigo-400 bg-zinc-900'
                : 'border-transparent text-zinc-400 hover:text-zinc-300 hover:bg-zinc-900/50'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        {/* Logic Tab */}
        {activeTab === 'logic' && (
          <div className="h-full flex flex-col">
            {/* Mermaid Diagram */}
            <div className="flex-1 min-h-0 border-b border-zinc-800 overflow-hidden">
              {mermaidCode ? (
                <div className="h-full overflow-auto">
                <MermaidRenderer code={mermaidCode} />
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-zinc-500">
                  <div className="text-center">
                    <Code2 className="w-12 h-12 mx-auto mb-4 text-zinc-600" />
                    <p className="text-zinc-500">No Mermaid diagram yet</p>
                    <p className="text-sm text-zinc-600 mt-2">Start a conversation to generate one</p>
                  </div>
                </div>
              )}
            </div>
            {/* Code Editor */}
            <div className="h-64 flex-shrink-0 border-t border-zinc-800 flex flex-col">
              <div className="px-4 py-2 border-b border-zinc-800 bg-zinc-900/50 flex-shrink-0">
                <span className="text-xs font-medium text-zinc-400">Mermaid Code</span>
              </div>
              <textarea
                value={mermaidCode}
                onChange={(e) => onMermaidCodeChange(e.target.value)}
                placeholder="graph LR&#10;&#10;A[Input] --> B[Process]&#10;B --> C[Output]"
                className="flex-1 w-full bg-zinc-900 text-zinc-100 font-mono text-sm p-4 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500/50 overflow-y-auto"
                style={{ fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace' }}
              />
            </div>

            {/* Logic Critic Section */}
            <div className="h-80 flex-shrink-0">
              {renderCriticSection(
                logicCritique,
                onLogicCritic,
                isGeneratingLogicCritic,
                !mermaidCode.trim(),
                'Please generate Mermaid code first'
              )}
            </div>
          </div>
        )}

        {/* Style Tab */}
        {activeTab === 'style' && (
          <div className="h-full flex flex-col">
            {/* Style Selector Section */}
            <div className="flex-shrink-0 px-4 py-4 border-b border-zinc-800 bg-zinc-900/50">
              <div className="mb-3">
                <span className="text-sm font-medium text-zinc-300">🎨 Visual Style</span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {/* AI Conference Style */}
                <button
                  onClick={() => onStyleChange('AI_CONFERENCE')}
                  className={`p-3 rounded-lg border-2 transition-all text-left ${
                    selectedStyle === 'AI_CONFERENCE'
                      ? 'border-indigo-500 bg-indigo-50/10'
                      : 'border-zinc-700 bg-zinc-800/50 hover:border-zinc-600'
                  }`}
                >
                  <div className="text-sm font-medium text-zinc-200 mb-1">
                    NeurIPS / CVPR / ICLR
                  </div>
                  <div className="text-xs text-zinc-400">
                    Abstract, isometric 2D, pastel colors
                  </div>
                </button>

                {/* Top Journal Style */}
                <button
                  onClick={() => onStyleChange('TOP_JOURNAL')}
                  className={`p-3 rounded-lg border-2 transition-all text-left ${
                    selectedStyle === 'TOP_JOURNAL'
                      ? 'border-indigo-500 bg-indigo-50/10'
                      : 'border-zinc-700 bg-zinc-800/50 hover:border-zinc-600'
                  }`}
                >
                  <div className="text-sm font-medium text-zinc-200 mb-1">
                    Nature / Science / Cell
                  </div>
                  <div className="text-xs text-zinc-400">
                    Hyper-realistic, dense, editorial
                  </div>
                </button>

                {/* Engineering Style */}
                <button
                  onClick={() => onStyleChange('ENGINEERING')}
                  className={`p-3 rounded-lg border-2 transition-all text-left ${
                    selectedStyle === 'ENGINEERING'
                      ? 'border-indigo-500 bg-indigo-50/10'
                      : 'border-zinc-700 bg-zinc-800/50 hover:border-zinc-600'
                  }`}
                >
                  <div className="text-sm font-medium text-zinc-200 mb-1">
                    IEEE / Industrial
                  </div>
                  <div className="text-xs text-zinc-400">
                    Technical blueprint, wireframe
                  </div>
                </button>

                {/* Custom Style */}
                <button
                  onClick={() => onStyleChange('CUSTOM')}
                  className={`p-3 rounded-lg border-2 transition-all text-left ${
                    selectedStyle === 'CUSTOM'
                      ? 'border-indigo-500 bg-indigo-50/10'
                      : 'border-zinc-700 bg-zinc-800/50 hover:border-zinc-600'
                  }`}
                >
                  <div className="text-sm font-medium text-zinc-200 mb-1">
                    ✨ Custom
                  </div>
                  <div className="text-xs text-zinc-400">
                    User defined style
                  </div>
                </button>
              </div>

              {/* Custom Style Input */}
              {selectedStyle === 'CUSTOM' && (
                <div className="mt-3">
                  <input
                    type="text"
                    value={customStyleText}
                    onChange={(e) => onCustomStyleTextChange(e.target.value)}
                    placeholder="e.g., Cyberpunk, Anime, Traditional Chinese Ink..."
                    className="w-full bg-zinc-800 text-zinc-100 text-sm px-3 py-2 rounded-lg border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500"
                  />
                </div>
              )}

              {/* Generate Schema Button */}
              <div className="mt-4 pt-4 border-t border-zinc-700">
                <button
                  onClick={onConfirmAndGenerateSchema}
                  disabled={!mermaidCode.trim() || isGeneratingSchema}
                  className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed text-white rounded-lg px-4 py-2.5 text-sm font-medium transition-all flex items-center justify-center gap-2"
                >
                  {isGeneratingSchema ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Generating Schema...
                    </>
                  ) : (
                    <>
                      <Palette className="w-4 h-4" />
                      Generate Visual Schema
                    </>
                  )}
                </button>
                {!mermaidCode.trim() && (
                  <p className="text-xs text-zinc-500 mt-2 text-center">
                    Please generate a Mermaid diagram in the Logic tab first
                  </p>
                )}
              </div>
            </div>

            {/* Visual Schema Editor */}
            <div className="flex-1 min-h-0 flex flex-col border-t border-zinc-800">
              <div className="px-4 py-2 border-b border-zinc-800 bg-zinc-900/50 flex items-center justify-between flex-shrink-0">
              <span className="text-xs font-medium text-zinc-400">Visual Schema</span>
                {visualSchema && (
                  <span className="text-xs text-zinc-500">Editable</span>
                )}
            </div>
            <textarea
              value={visualSchema}
              onChange={(e) => onVisualSchemaChange(e.target.value)}
              placeholder="[Style & Meta-Instructions]&#10;High-fidelity scientific schematic, clean white background...&#10;&#10;[LAYOUT CONFIGURATION]&#10;* Selected Layout: Linear..."
                className="flex-1 w-full bg-zinc-900 text-zinc-100 text-sm p-4 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500/50 overflow-y-auto"
            />
            </div>

            {/* Style Critic Section */}
            <div className="h-80 flex-shrink-0">
              {renderCriticSection(
                styleCritique,
                onStyleCritic,
                isGeneratingStyleCritic,
                !visualSchema.trim(),
                'Please generate Visual Schema first'
              )}
            </div>
          </div>
        )}

        {/* Result Tab */}
        {activeTab === 'result' && (
          <div className="h-full flex flex-col">
            {/* Image Display Section */}
            <div className="flex-1 min-h-0 flex flex-col border-b border-zinc-800">
              <div className="px-4 py-2 border-b border-zinc-800 bg-zinc-900/50 flex items-center justify-between flex-shrink-0">
                <span className="text-xs font-medium text-zinc-400">Generated Image</span>
                {imageUrl && onDownloadImage && (
                  <button
                    onClick={onDownloadImage}
                    className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 transition-colors"
                  >
                    <Download className="w-3 h-3" />
                    Download
                  </button>
                )}
              </div>
              <div className="flex-1 min-h-0 p-6 flex items-center justify-center bg-zinc-900/30 overflow-auto">
              {imageUrl ? (
                <div className="max-w-full max-h-full">
                  {imageUrl.startsWith('data:') ? (
                    <img
                      src={imageUrl}
                      alt="Generated figure"
                      className="max-w-full max-h-full object-contain rounded-lg border border-zinc-800 shadow-lg"
                    />
                  ) : (
                    <img
                      src={imageUrl}
                      alt="Generated figure"
                      className="max-w-full max-h-full object-contain rounded-lg border border-zinc-800 shadow-lg"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                        const parent = target.parentElement;
                        if (parent) {
                          parent.innerHTML = `
                            <div class="text-zinc-500 text-center p-8">
                              <p class="text-sm">Failed to load image</p>
                              <p class="text-xs text-zinc-600 mt-2">URL: ${imageUrl.substring(0, 50)}...</p>
                            </div>
                          `;
                        }
                      }}
                    />
                  )}
                </div>
              ) : (
                <div className="text-center text-zinc-500">
                  <ImageIcon className="w-12 h-12 mx-auto mb-4 text-zinc-600" />
                  <p className="text-zinc-500">No image generated yet</p>
                  <p className="text-sm text-zinc-600 mt-2">Generate a schema and render it</p>
                </div>
                )}
              </div>
            </div>

            {/* Result Critic Section */}
            <div className="h-80 flex-shrink-0">
              {renderCriticSection(
                resultCritique,
                onResultCritic,
                isGeneratingResultCritic,
                !imageUrl,
                'Please generate an image first'
              )}
            </div>

            {/* Action Buttons */}
            <div className="px-4 py-3 border-t border-zinc-800 bg-zinc-900/50 space-y-2">
              {/* Start Rendering Button */}
              <button
                onClick={onStartRendering}
                disabled={!visualSchema.trim() || isRendering}
                className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed text-white rounded-lg px-4 py-2.5 text-sm font-medium transition-all flex items-center justify-center gap-2"
              >
                {isRendering ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Rendering...
                  </>
                  ) : (
                  <>
                    <ImageIcon className="w-4 h-4" />
                    Start Rendering
                  </>
                  )}
              </button>
              {!visualSchema.trim() && (
                <p className="text-xs text-zinc-500 text-center">
                  Please generate a Visual Schema in the Style tab first
                </p>
            )}

              {/* Regenerate Button */}
            {imageUrl && (
                <button
                  onClick={onRegenerate}
                  disabled={isRendering || !visualSchema.trim()}
                  className="w-full bg-zinc-700 hover:bg-zinc-600 disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed text-white rounded-lg px-4 py-2 text-sm font-medium transition-all flex items-center justify-center gap-2"
                >
                  {isRendering ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Regenerating...
                    </>
                  ) : (
                    <>
                      <ImageIcon className="w-4 h-4" />
                      Regenerate
                    </>
                  )}
                </button>
              )}
              </div>
          </div>
        )}

        {/* Figure to PPT Tab */}
        {activeTab === 'figure-to-ppt' && (
          <div className="h-full overflow-hidden">
            <FigureToPPT />
          </div>
        )}
      </div>
    </div>
  );
};

export default WorkspaceTabs;

