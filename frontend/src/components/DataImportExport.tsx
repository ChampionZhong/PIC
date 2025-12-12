import { useRef, useState } from 'react';
import { Download, Upload, Check, AlertCircle } from 'lucide-react';
import { Conversation } from '../types';
import { exportConversations, importConversations } from '../utils/dataExport';

interface DataImportExportProps {
  conversations: Conversation[];
  onImport: (conversations: Conversation[]) => void;
}

const DataImportExport: React.FC<DataImportExportProps> = ({
  conversations,
  onImport,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importStatus, setImportStatus] = useState<{
    type: 'success' | 'error' | null;
    message: string;
  }>({ type: null, message: '' });

  const handleExport = () => {
    try {
      exportConversations(conversations);
      setImportStatus({
        type: 'success',
        message: '导出成功！',
      });
      setTimeout(() => setImportStatus({ type: null, message: '' }), 3000);
    } catch (error) {
      setImportStatus({
        type: 'error',
        message: error instanceof Error ? error.message : '导出失败',
      });
      setTimeout(() => setImportStatus({ type: null, message: '' }), 5000);
    }
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    try {
      const imported = await importConversations(file);
      
      // Ask user if they want to merge or replace
      const shouldMerge = window.confirm(
        `成功导入 ${imported.length} 个对话。\n\n` +
        `点击"确定"合并到现有对话，点击"取消"替换所有对话。`
      );

      if (shouldMerge) {
        // Merge: add imported conversations, avoiding ID conflicts
        const existingIds = new Set(conversations.map((c) => c.id));
        const merged = [
          ...conversations,
          ...imported.map((conv) => {
            // If ID conflicts, generate new ID
            if (existingIds.has(conv.id)) {
              return {
                ...conv,
                id: `conv-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
              };
            }
            return conv;
          }),
        ];
        onImport(merged);
        setImportStatus({
          type: 'success',
          message: `成功合并 ${imported.length} 个对话！`,
        });
      } else {
        // Replace: use imported conversations
        onImport(imported);
        setImportStatus({
          type: 'success',
          message: `成功导入 ${imported.length} 个对话，已替换现有对话！`,
        });
      }

      setTimeout(() => setImportStatus({ type: null, message: '' }), 5000);
    } catch (error) {
      setImportStatus({
        type: 'error',
        message: error instanceof Error ? error.message : '导入失败',
      });
      setTimeout(() => setImportStatus({ type: null, message: '' }), 5000);
    }
  };

  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-800 bg-slate-900/50">
      <div className="flex items-center gap-2 flex-1">
        <button
          onClick={handleExport}
          className="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 flex items-center gap-1.5 transition-colors"
          title="导出所有对话数据"
        >
          <Download className="w-3 h-3" />
          <span>导出</span>
        </button>
        <button
          onClick={handleImportClick}
          className="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 flex items-center gap-1.5 transition-colors"
          title="导入对话数据"
        >
          <Upload className="w-3 h-3" />
          <span>导入</span>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
      {importStatus.type && (
        <div
          className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded ${
            importStatus.type === 'success'
              ? 'bg-green-900/50 text-green-300 border border-green-800'
              : 'bg-red-900/50 text-red-300 border border-red-800'
          }`}
        >
          {importStatus.type === 'success' ? (
            <Check className="w-3 h-3" />
          ) : (
            <AlertCircle className="w-3 h-3" />
          )}
          <span>{importStatus.message}</span>
        </div>
      )}
    </div>
  );
};

export default DataImportExport;

