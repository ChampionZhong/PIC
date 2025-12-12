export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface MermaidCodeTab {
  id: number;
  label: string;
  code: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  mermaidTabs: MermaidCodeTab[];
  activeTabId: number | null;
  tabCounter: number;
  currentCode: string;
  inputDraft: string; // 保存用户输入的草稿
  createdAt: number;
  updatedAt: number;
}

