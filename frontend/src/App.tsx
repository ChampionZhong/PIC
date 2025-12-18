import { useState } from 'react';
import ChatPanel, { ChatMessage, ChatSender } from './components/ChatPanel';
import WorkspaceTabs, { ActiveTab } from './components/WorkspaceTabs';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  // Global State Management
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [activeTab, setActiveTab] = useState<ActiveTab>('logic');
  const [mermaidCode, setMermaidCode] = useState<string>('');
  const [visualSchema, setVisualSchema] = useState<string>('');
  const [imageUrl, setImageUrl] = useState<string>('');
  const [critique, setCritique] = useState<string>('');
  
  // Critic states for each tab
  const [logicCritique, setLogicCritique] = useState<{feedback: string; passed: boolean; suggestions?: string} | null>(null);
  const [styleCritique, setStyleCritique] = useState<{feedback: string; passed: boolean; suggestions?: string} | null>(null);
  const [resultCritique, setResultCritique] = useState<{feedback: string; passed: boolean; suggestions?: string} | null>(null);
  
  // Loading states
  const [isGeneratingSchema, setIsGeneratingSchema] = useState(false);
  const [isRendering, setIsRendering] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isGeneratingLogicCritic, setIsGeneratingLogicCritic] = useState(false);
  const [isGeneratingStyleCritic, setIsGeneratingStyleCritic] = useState(false);
  const [isGeneratingResultCritic, setIsGeneratingResultCritic] = useState(false);
  const [originalIdea, setOriginalIdea] = useState<string>('');
  
  // Style selection state
  const [selectedStyle, setSelectedStyle] = useState<string>('AI_CONFERENCE');
  const [customStyleText, setCustomStyleText] = useState<string>('');

  // Add message to chat history
  const addChatMessage = (sender: ChatSender, message: string) => {
    setChatHistory((prev) => [
      ...prev,
      { sender, message, timestamp: Date.now() },
    ]);
  };

  // Handle user message - call unified Manager-driven Chat API
  const handleSendMessage = async (message: string) => {
    if (!message.trim()) return;
    
    // Add user message to chat
    addChatMessage('User', message);
    if (!originalIdea) {
      setOriginalIdea(message);
    }
    setIsLoading(true);

    try {
      // Build chat history for API (convert ChatMessage[] to API format)
      const apiHistory = chatHistory.map(msg => ({
        role: msg.sender === 'User' ? 'user' : 'assistant',
        content: msg.message
      }));

      // Build current artifacts
      const currentArtifacts: Record<string, string> = {};
      if (mermaidCode) currentArtifacts.mermaid_code = mermaidCode;
      if (visualSchema) currentArtifacts.visual_schema = visualSchema;
      if (imageUrl) currentArtifacts.image_url = imageUrl;

      // Build request body with style parameters
      const requestBody: any = {
        message: message,
        history: apiHistory,
        active_tab: activeTab,
        current_artifacts: currentArtifacts,
        style_mode: selectedStyle,
      };
      
      if (selectedStyle === 'CUSTOM' && customStyleText.trim()) {
        requestBody.custom_style_prompt = customStyleText.trim();
      }

      // Call unified /api/chat endpoint
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || 'Failed to process request'}`);
      }

      const data = await response.json();
      
      // Display Manager reasoning as a system note
      if (data.manager_reasoning) {
        addChatMessage('Manager', `Routing to ${data.agent_name.replace('_', ' ')}: ${data.manager_reasoning}`);
      }
      
      // Map agent_name to ChatSender
      const agentSenderMap: Record<string, ChatSender> = {
        'architect': 'Architect',
        'art_director': 'Director',
        'painter': 'Painter',
        'critic': 'Critic',
      };
      
      const agentSender = agentSenderMap[data.agent_name] || 'Architect';
      
      // Add agent's response to chat
      if (data.response_text) {
        addChatMessage(agentSender, data.response_text);
      }
      
      // Handle updated artifacts and switch tabs
      if (data.updated_artifacts) {
        // Update mermaid_code -> Switch to Logic Tab
        if (data.updated_artifacts.mermaid_code) {
          setMermaidCode(data.updated_artifacts.mermaid_code);
          setActiveTab('logic');
        }
        
        // Update visual_schema -> Switch to Style Tab
        if (data.updated_artifacts.visual_schema) {
          setVisualSchema(data.updated_artifacts.visual_schema);
          setActiveTab('style');
        }
        
        // Update image_url -> Switch to Result Tab
        if (data.updated_artifacts.image_url) {
          setImageUrl(data.updated_artifacts.image_url);
          setActiveTab('result');
          // Note: Critic review is now handled manually via the Result tab's Critic section
        }
        
        // Update critique
        if (data.updated_artifacts.critique) {
          setCritique(data.updated_artifacts.critique);
        }
      }
    } catch (error) {
      console.error('Error calling chat API:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to process request';
      addChatMessage('Manager', `Error: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle "Confirm & Generate Schema" - use unified API
  const handleConfirmAndGenerateSchema = async () => {
    if (!mermaidCode.trim()) return;

    setIsGeneratingSchema(true);
    addChatMessage('User', 'Generate visual schema from this Mermaid diagram');

    try {
      // Build chat history for API
      const apiHistory = chatHistory.map(msg => ({
        role: msg.sender === 'User' ? 'user' : 'assistant',
        content: msg.message
      }));

      // Build current artifacts
      const currentArtifacts: Record<string, string> = {};
      currentArtifacts.mermaid_code = mermaidCode;
      if (visualSchema) currentArtifacts.visual_schema = visualSchema;
      if (imageUrl) currentArtifacts.image_url = imageUrl;

      // Use unified API with style parameters
      const requestBody: any = {
        message: 'Generate visual schema from this Mermaid diagram',
        history: apiHistory,
        active_tab: 'logic',
        current_artifacts: currentArtifacts,
        style_mode: selectedStyle,
      };
      
      if (selectedStyle === 'CUSTOM' && customStyleText.trim()) {
        requestBody.custom_style_prompt = customStyleText.trim();
      }

      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || 'Failed to generate schema'}`);
      }

      const data = await response.json();
      
      // Display Manager reasoning if present
      if (data.manager_reasoning) {
        addChatMessage('Manager', `Routing to ${data.agent_name.replace('_', ' ')}: ${data.manager_reasoning}`);
      }
      
      // Handle updated artifacts
      if (data.updated_artifacts?.visual_schema) {
        setVisualSchema(data.updated_artifacts.visual_schema);
        setActiveTab('style');
        addChatMessage('Director', data.response_text);
      } else if (data.response_text) {
        addChatMessage('Director', data.response_text);
      } else {
        throw new Error('No visual schema in response');
      }
    } catch (error) {
      console.error('Error calling Art Director agent:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate schema';
      addChatMessage('Director', `Error: ${errorMessage}`);
    } finally {
      setIsGeneratingSchema(false);
    }
  };

  // Handle "Start Rendering" - use unified API
  const handleStartRendering = async () => {
    if (!visualSchema.trim()) return;

    setIsRendering(true);
    addChatMessage('User', 'Render the image from this visual schema');

    try {
      // Build chat history for API
      const apiHistory = chatHistory.map(msg => ({
        role: msg.sender === 'User' ? 'user' : 'assistant',
        content: msg.message
      }));

      // Build current artifacts
      const currentArtifacts: Record<string, string> = {};
      if (mermaidCode) currentArtifacts.mermaid_code = mermaidCode;
      currentArtifacts.visual_schema = visualSchema;
      if (imageUrl) currentArtifacts.image_url = imageUrl;

      // Use unified API
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: 'Render the image from this visual schema',
          history: apiHistory,
          active_tab: 'style',
          current_artifacts: currentArtifacts,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || 'Failed to render image'}`);
      }

      const data = await response.json();
      
      // Display Manager reasoning if present
      if (data.manager_reasoning) {
        addChatMessage('Manager', `Routing to ${data.agent_name.replace('_', ' ')}: ${data.manager_reasoning}`);
      }
      
      // Handle updated artifacts
      if (data.updated_artifacts?.image_url) {
        setImageUrl(data.updated_artifacts.image_url);
        setActiveTab('result');
        addChatMessage('Painter', data.response_text);
        // Note: Critic review is now handled manually via the Result tab's Critic section
      } else if (data.response_text) {
        addChatMessage('Painter', data.response_text);
      } else {
        throw new Error('No image URL in response');
      }
    } catch (error) {
      console.error('Error calling Painter agent:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to render image';
      addChatMessage('Painter', `Error: ${errorMessage}`);
    } finally {
      setIsRendering(false);
    }
  };

  // Handle Critic review (Auto-triggered after image generation)
  const handleCriticReview = async (imageUrlToReview: string) => {
    if (!imageUrlToReview || !originalIdea) return;

    addChatMessage('Critic', 'Reviewing generated image...');

    try {
      // Build chat history for API
      const apiHistory = chatHistory.map(msg => ({
        role: msg.sender === 'User' ? 'user' : 'assistant',
        content: msg.message
      }));

      // Build current artifacts
      const currentArtifacts: Record<string, string> = {};
      if (mermaidCode) currentArtifacts.mermaid_code = mermaidCode;
      if (visualSchema) currentArtifacts.visual_schema = visualSchema;
      currentArtifacts.image_url = imageUrlToReview;

      // Use unified API for critic review
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: `Please review the generated image against the original idea: ${originalIdea}`,
          history: apiHistory,
          active_tab: 'result',
          current_artifacts: currentArtifacts,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || 'Failed to review image'}`);
      }

      const data = await response.json();
      
      // Display Manager reasoning if present
      if (data.manager_reasoning) {
        addChatMessage('Manager', `Routing to ${data.agent_name.replace('_', ' ')}: ${data.manager_reasoning}`);
      }
      
      // Display the critique
      if (data.updated_artifacts?.critique) {
        setCritique(data.updated_artifacts.critique);
        addChatMessage('Critic', data.response_text);
      } else if (data.response_text) {
        addChatMessage('Critic', data.response_text);
      } else {
        throw new Error('No feedback in response');
      }
    } catch (error) {
      console.error('Error calling Critic agent:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to review image';
      addChatMessage('Critic', `Error: ${errorMessage}`);
      setCritique(`Error: ${errorMessage}`);
    }
  };

  // Handle "Regenerate" - re-render with current schema
  const handleRegenerate = async () => {
    if (!visualSchema.trim()) return;
    await handleStartRendering();
  };

  // Handle image download
  const handleDownloadImage = () => {
    if (!imageUrl) return;

    try {
      if (imageUrl.startsWith('data:')) {
        // Handle base64 data URL
        const link = document.createElement('a');
        link.href = imageUrl;
        link.download = 'generated-figure.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } else {
        // Handle HTTP URL
        const link = document.createElement('a');
        link.href = imageUrl;
        link.download = 'generated-figure.png';
        link.target = '_blank';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (error) {
      console.error('Error downloading image:', error);
      // Fallback: open in new tab
      window.open(imageUrl, '_blank');
    }
  };

  // Handle Logic Critic
  const handleLogicCritic = async () => {
    if (!mermaidCode.trim()) return;

    setIsGeneratingLogicCritic(true);
    setLogicCritique(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/agent/critic/logic`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mermaid_code: mermaidCode,
          original_idea: originalIdea || undefined,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || 'Failed to generate logic critique'}`);
      }

      const data = await response.json();
      setLogicCritique({
        feedback: data.feedback || '',
        passed: data.passed || false,
        suggestions: data.suggestions || undefined,
      });
    } catch (error) {
      console.error('Error calling Logic Critic:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate critique';
      setLogicCritique({
        feedback: `Error: ${errorMessage}`,
        passed: false,
      });
    } finally {
      setIsGeneratingLogicCritic(false);
    }
  };

  // Handle Style Critic
  const handleStyleCritic = async () => {
    if (!visualSchema.trim()) return;

    setIsGeneratingStyleCritic(true);
    setStyleCritique(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/agent/critic/style`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          visual_schema: visualSchema,
          mermaid_code: mermaidCode || undefined,
          style_mode: selectedStyle || undefined,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || 'Failed to generate style critique'}`);
      }

      const data = await response.json();
      setStyleCritique({
        feedback: data.feedback || '',
        passed: data.passed || false,
        suggestions: data.suggestions || undefined,
      });
    } catch (error) {
      console.error('Error calling Style Critic:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate critique';
      setStyleCritique({
        feedback: `Error: ${errorMessage}`,
        passed: false,
      });
    } finally {
      setIsGeneratingStyleCritic(false);
    }
  };

  // Handle Result Critic
  const handleResultCritic = async () => {
    if (!imageUrl) return;

    setIsGeneratingResultCritic(true);
    setResultCritique(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/agent/critic/result`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_url: imageUrl,
          original_idea: originalIdea || 'Generated scientific figure',
          visual_schema: visualSchema || undefined,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || 'Failed to generate result critique'}`);
      }

      const data = await response.json();
      setResultCritique({
        feedback: data.feedback || '',
        passed: data.passed || false,
        suggestions: data.suggestions || undefined,
      });
    } catch (error) {
      console.error('Error calling Result Critic:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate critique';
      setResultCritique({
        feedback: `Error: ${errorMessage}`,
        passed: false,
      });
    } finally {
      setIsGeneratingResultCritic(false);
    }
  };

  return (
    <div className="h-screen w-full flex bg-zinc-950 overflow-hidden">
      {/* Left Panel - Chat Interface (35%) */}
      <div className="w-[35%] flex-shrink-0">
        <ChatPanel
          chatHistory={chatHistory}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      </div>

      {/* Right Panel - Tabbed Workspace (65%) */}
      <div className="flex-1 min-w-0">
        <WorkspaceTabs
          activeTab={activeTab}
          onTabChange={setActiveTab}
          mermaidCode={mermaidCode}
          onMermaidCodeChange={setMermaidCode}
          visualSchema={visualSchema}
          onVisualSchemaChange={setVisualSchema}
          imageUrl={imageUrl}
          critique={critique}
          onConfirmAndGenerateSchema={handleConfirmAndGenerateSchema}
          onStartRendering={handleStartRendering}
          onRegenerate={handleRegenerate}
          isGeneratingSchema={isGeneratingSchema}
          isRendering={isRendering}
          selectedStyle={selectedStyle}
          onStyleChange={setSelectedStyle}
          customStyleText={customStyleText}
          onCustomStyleTextChange={setCustomStyleText}
          onDownloadImage={handleDownloadImage}
          logicCritique={logicCritique}
          styleCritique={styleCritique}
          resultCritique={resultCritique}
          onLogicCritic={handleLogicCritic}
          onStyleCritic={handleStyleCritic}
          onResultCritic={handleResultCritic}
          isGeneratingLogicCritic={isGeneratingLogicCritic}
          isGeneratingStyleCritic={isGeneratingStyleCritic}
          isGeneratingResultCritic={isGeneratingResultCritic}
        />
      </div>
    </div>
  );
}

export default App;
