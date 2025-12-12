import { Conversation } from '../types';

/**
 * Export all conversations to a JSON file
 */
export const exportConversations = (conversations: Conversation[]): void => {
  try {
    const dataStr = JSON.stringify(conversations, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `research-copilot-backup-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Failed to export conversations:', error);
    throw new Error('导出失败，请重试');
  }
};

/**
 * Import conversations from a JSON file
 */
export const importConversations = async (file: File): Promise<Conversation[]> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const imported = JSON.parse(content);
        
        // Validate imported data
        if (!Array.isArray(imported)) {
          throw new Error('导入文件格式不正确：必须是对话数组');
        }
        
        // Validate each conversation structure
        const validated: Conversation[] = imported.map((conv: any, index: number) => {
          if (!conv.id || !conv.title) {
            throw new Error(`对话 ${index + 1} 缺少必要字段（id 或 title）`);
          }
          
          return {
            id: conv.id || `conv-${Date.now()}-${index}`,
            title: conv.title || `Imported Conversation ${index + 1}`,
            messages: Array.isArray(conv.messages) ? conv.messages : [],
            mermaidTabs: Array.isArray(conv.mermaidTabs) ? conv.mermaidTabs : [],
            activeTabId: conv.activeTabId || null,
            tabCounter: typeof conv.tabCounter === 'number' ? conv.tabCounter : 1,
            currentCode: conv.currentCode || '',
            createdAt: typeof conv.createdAt === 'number' ? conv.createdAt : Date.now(),
            updatedAt: typeof conv.updatedAt === 'number' ? conv.updatedAt : Date.now(),
          };
        });
        
        resolve(validated);
      } catch (error) {
        console.error('Failed to import conversations:', error);
        reject(error instanceof Error ? error : new Error('导入失败：文件格式不正确'));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('读取文件失败'));
    };
    
    reader.readAsText(file);
  });
};

