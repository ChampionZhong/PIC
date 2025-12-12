"""
FastAPI backend for AI Research Figure Generator
"""
import os
import base64
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# Try to import Google Generative AI for Gemini support
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Research Figure Generator API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get OpenAI configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize OpenAI client
client_config = {"api_key": OPENAI_API_KEY}
if OPENAI_API_URL:
    client_config["base_url"] = OPENAI_API_URL

openai_client = OpenAI(**client_config)

# Get Gemini configuration from environment (optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")  # Custom API URL if provided (OpenAI-compatible)

# Initialize Gemini client (use OpenAI-compatible client if custom URL is provided)
gemini_client = None
if GEMINI_API_KEY:
    if GEMINI_API_URL:
        # Use OpenAI-compatible client for custom Gemini API endpoint
        gemini_client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_API_URL
        )
    elif GEMINI_AVAILABLE:
        # Use native Google Generative AI library
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = "native"  # Marker to use native library

# System prompt for Research Architect
RESEARCH_ARCHITECT_SYSTEM_PROMPT = """You are the "Research Architect & Visual Director".

Goal: Transform a vague idea into a Top-tier Conference Main Figure (Figure 1) structure.

CRITICAL VISUAL CONSTRAINT: A good Figure 1 is ABSTRACT and CLEAN. It should not look like a software engineering flowchart.

Max Nodes: Aim for 5-9 high-level blocks total.

Grouping: Encapsulate detailed steps (e.g., "Tokenization", "Embedding", "Vector Look-up") into a single high-level Subgraph (e.g., "Retrieval Module").

Hiding Details: Do not show utility nodes like "Config", "Logging", or "Database Connection". Only show the Data Flow.

Emphasis: Explicitly use styles to highlight the Novelty (the part the user invented).

Workflow:

Analyze: Infer missing components based on domain knowledge.

Abstract: Group low-level operations into high-level "Phases" or "Modules".

Visualize: Generate mermaid code that represents this high-level view.

Output Format: [[ANALYSIS_START]] ... (Explain your abstraction choices: "I grouped the text processing steps into a single 'Parser' module to keep the figure clean.") [[ANALYSIS_END]]

[[MERMAID_START]] graph LR %% Use subgraphs for high-level grouping subgraph "Phase 1: ..." NodeA end %% ... [[MERMAID_END]]

**CRITICAL Mermaid Code Rules (MUST FOLLOW):**

1. **Graph Declaration:**
   - ALWAYS put `graph LR` (or TB/TD/BT/RL) on its own line
   - ALWAYS add a blank line after the graph declaration
   - Example:
     ```
     graph LR

     A[Node] --> B[Node]
     ```
   - NOT: `graph LRA[Node]` (wrong - no newline)

2. **Node Labels:**
   - NEVER use HTML tags: NO <br/>, <p>, <div>, etc.
   - NEVER use newline characters (\n) inside node labels
   - Use spaces for multi-word labels: `[Painter Agent LLM code generator]`
   - NOT: `[Painter Agent LLM\ncode generator]` (wrong - has \n)
   - NOT: `[Painter Agent LLM<br/>code generator]` (wrong - has HTML)

3. **Edge Labels:**
   - Same rules as node labels: NO HTML, NO newlines
   - Use spaces: `A -->|Upload PDF paper Poster constraints| B`
   - NOT: `A -->|Upload PDF paper\nPoster constraints| B` (wrong)

4. **Comments:**
   - Keep comments simple: `%% Stage 1: Parsing`
   - NO decorative separators: NO `%% ============`, NO `%% ----------`
   - NOT: `%% ============ PARSING STAGE ============` (wrong)

5. **Code Structure:**
   - Use proper indentation for subgraphs
   - Keep each node definition on its own line
   - Use blank lines to separate logical sections

**Example of CORRECT Mermaid Code:**

```
graph LR

%% High-level flow
A[Input PDF Paper] --> B[Parsing Stage]
B --> C[Planning Stage]
C --> D[Refinement Stage]

%% Parsing details
subgraph S1[Parsing Stage]
  B1[PDF Loader] --> B2[PDF Parser text and figures]
  B2 --> B3[Section Segmenter]
end
```

**Example of WRONG Mermaid Code (DO NOT GENERATE):**

```
graph LRA[Input PDF Paper]  # WRONG: no newline after graph LR
A[PDF Parser<br/>text extract]  # WRONG: HTML tag
B[Section\nSummarizer]  # WRONG: newline in label
%% ============ PARSING ============  # WRONG: decorative separators
```

**Remember:** Generate clean, valid Mermaid code that can be rendered directly without modification."""

# System prompt for Art Director
ART_DIRECTOR_SYSTEM_PROMPT = """
# Role
You are the **Art Director** for a top-tier AI conference (NeurIPS/CVPR). You work with a Logic Engineer who provides you with a system topology (Mermaid Code).

# Your Task
Translate the abstract **Mermaid Code** into a concrete **[VISUAL SCHEMA]**. This schema will be used by an AI illustrator (Gemini 3) to generate the final Main Figure.

# Input
1.  **Mermaid Code:** The strict logic and data flow.
2.  **Context:** (Optional) A brief description of the paper's domain (e.g., "This is a Multi-modal Agent system").

# Critical Transformation Rules (The "Magic")
Since you don't have the full paper text, you must use **Domain Knowledge Inference** to visualize abstract nodes:
1.  **Visual Metaphors:**
    * If Mermaid says `Parser`, visualize it as "A document being scanned by a ray of light".
    * If Mermaid says `Neural Network`, visualize it as "3D stacked glass layers".
    * If Mermaid says `Database`, visualize it as "A cylindrical data silo".
2.  **Layout Interpretation:**
    * If the Mermaid has a feedback loop, force the Layout to be `Cyclic/Iterative`.
    * If the Mermaid is `Input -> Process -> Output`, use `Linear Pipeline`.
3.  **Grouping:** Map Mermaid `subgraphs` directly to Visual `Zones`.

# Output Format
(Strictly output the schema block below. Do not output markdown code fences around it, just raw text.)

---BEGIN PROMPT---

[Style & Meta-Instructions]
High-fidelity scientific schematic, technical vector illustration, clean white background, distinct boundaries, academic textbook style. High resolution 4k, strictly 2D flat design with subtle isometric elements. Soft lighting, academic pastel colors.

[LAYOUT CONFIGURATION]
* **Selected Layout**: [Choose based on Mermaid structure: Linear / Cyclic / Hierarchical]
* **Composition Logic**: [Briefly describe how the Mermaid nodes are arranged spatially]
* **Color Palette**: [e.g., Azure Blue (Logic), Coral Orange (Novelty), Slate Grey (Data)]

[ZONE 1: {Mermaid Node/Subgraph Name}]
* **Container**: [e.g., Left-side Panel]
* **Visual Structure**: [Describe the visual metaphor. e.g., "A stack of PDF icons transforming into floating text blocks"]
* **Key Text Labels**: [Extract EXACT text from Mermaid node labels]

[ZONE 2: {Mermaid Node/Subgraph Name}]
* **Container**: [e.g., Central Processing Unit]
* **Visual Structure**: [Describe the visual metaphor. e.g., "A glowing binary tree structure growing out of a digital grid"]
* **Key Text Labels**: [Extract EXACT text from Mermaid node labels]

... (Generate Zones for all key Mermaid components)

[CONNECTIONS]
1. [Describe edges based on Mermaid arrows. e.g., "A wide blue arrow flowing from Zone 1 to Zone 2"]
2. [CRITICAL: If Mermaid has a loop, describe it as "A curved red dashed arrow looping back..."]

---END PROMPT---
"""


# Pydantic models
class UserRequest(BaseModel):
    message: str
    history: List[dict] = []


class ChatResponse(BaseModel):
    response: str


class GenerateImageRequest(BaseModel):
    mermaid_code: str
    style: str = "neurips"


class RenderImageResponse(BaseModel):
    visual_schema: str = Field(alias="schema", serialization_alias="schema")
    image_url: str = Field(alias="image_url", serialization_alias="imageUrl")
    
    model_config = {"populate_by_name": True}


def build_messages(history: List[dict], current_message: str) -> List[dict]:
    """
    Build the messages list for OpenAI API from history and current message.
    
    Args:
        history: List of previous conversation messages
        current_message: The current user message
    
    Returns:
        List of message dictionaries formatted for OpenAI API
    """
    messages = [{"role": "system", "content": RESEARCH_ARCHITECT_SYSTEM_PROMPT}]
    
    # Add history messages
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Add current user message
    messages.append({"role": "user", "content": current_message})
    
    return messages


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: UserRequest):
    """
    POST endpoint for chat interaction with Research Architect.
    
    Args:
        request: UserRequest containing message and conversation history
    
    Returns:
        ChatResponse containing the raw AI response
    """
    try:
        # Build messages for OpenAI API
        messages = build_messages(request.history, request.message)
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-5.1",  # Default model, can be overridden via env if needed
            messages=messages,
            temperature=0.0,
        )
        
        # Extract the assistant's response
        ai_response = response.choices[0].message.content
        
        return ChatResponse(response=ai_response)
    
    except Exception as e:
        # Handle all errors
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "AI Research Figure Generator API is running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


def generate_visual_schema(mermaid_code: str) -> str:
    """
    The "Art Director" Service: Transforms Mermaid code into a Visual Schema.
    
    Args:
        mermaid_code: The Mermaid diagram code to transform
        
    Returns:
        The visual schema as a structured text description
        
    Raises:
        HTTPException: If the LLM call fails
    """
    try:
        # Build messages for OpenAI API
        messages = [
            {"role": "system", "content": ART_DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate this Mermaid code into a concise Visual Schema (under 2000 characters):\n\n{mermaid_code}"}
        ]
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-5.1",  # Using the same model as chat endpoint
            messages=messages,
            temperature=0.7,  # Slightly higher temperature for creative schema generation
        )
        
        # Extract the visual schema
        visual_schema = response.choices[0].message.content.strip()
        
        # Extract schema from markers if present
        if "---BEGIN PROMPT---" in visual_schema and "---END PROMPT---" in visual_schema:
            start_idx = visual_schema.find("---BEGIN PROMPT---") + len("---BEGIN PROMPT---")
            end_idx = visual_schema.find("---END PROMPT---")
            visual_schema = visual_schema[start_idx:end_idx].strip()
        
        return visual_schema
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating visual schema: {str(e)}"
        )


def render_image(visual_schema: str, style: str = "neurips") -> str:
    """
    The "Painter" Service: Generates an image from a visual schema using Gemini 3 or DALL-E 3.
    
    Args:
        visual_schema: The visual schema description
        style: The style to use (default: "neurips")
        
    Returns:
        URL or base64 data URL of the generated image
        
    Raises:
        HTTPException: If the image generation fails
    """
    # Wrap the schema with detailed style instructions and prompt wrapper
    prompt_template = """**Style Reference & Execution Instructions:**

1.  **Art Style (Visio/Illustrator Aesthetic):**

    Generate a **professional academic architecture diagram** suitable for a top-tier computer science paper (CVPR/NeurIPS).

    *  **Visuals:**  Flat vector graphics, distinct geometric shapes, clean thin outlines, and soft pastel fills (Azure Blue, Slate Grey, Coral Orange).

    *  **Layout:**  Strictly follow the spatial arrangement defined below.

    *  **Vibe:**  Technical, precise, clean white background. NOT hand-drawn, NOT photorealistic, NOT 3D render, NO shadows/shading.



2.  **CRITICAL TEXT CONSTRAINTS (Read Carefully):**

    *  **DO NOT render meta-labels:**  Do not write words like "ZONE 1", "LAYOUT CONFIGURATION", "Input", "Output", or "Container" inside the image. These are structural instructions for YOU, not text for the image.

    *  **ONLY render "Key Text Labels":**  Only text inside double quotes (e.g., "[Text]") listed under "Key Text Labels" should appear in the diagram.

    *  **Font:**  Use a clean, bold Sans-Serif font (like Roboto or Helvetica) for all labels.



3.  **Visual Schema Execution:**

    Translate the following structural blueprint into the final image:

{visual_schema}"""
    
    # Calculate wrapper text length (without visual_schema)
    wrapper_length = len(prompt_template.replace('{visual_schema}', ''))
    MAX_TOTAL_LENGTH = 4000  # API limit
    MAX_SCHEMA_LENGTH = MAX_TOTAL_LENGTH - wrapper_length - 100  # Leave 100 chars buffer
    
    # Truncate visual_schema if too long
    if len(visual_schema) > MAX_SCHEMA_LENGTH:
        # Try to truncate at a sentence boundary for better readability
        truncated = visual_schema[:MAX_SCHEMA_LENGTH]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        # Use the later of period or newline if found in last 20% of text
        cutoff_point = max(last_period, last_newline) if max(last_period, last_newline) > MAX_SCHEMA_LENGTH * 0.8 else MAX_SCHEMA_LENGTH
        if cutoff_point > MAX_SCHEMA_LENGTH * 0.8:
            visual_schema = truncated[:cutoff_point + 1] + "\n[Schema truncated for length]"
        else:
            visual_schema = truncated + "... [Schema truncated for length]"
    
    # Build the final prompt
    prompt = prompt_template.format(visual_schema=visual_schema)
    
    # Final safety check: truncate prompt if still too long (shouldn't happen, but safety net)
    if len(prompt) > MAX_TOTAL_LENGTH:
        prompt = prompt[:MAX_TOTAL_LENGTH - 50] + "... [Prompt truncated]"
    
    try:
        # Try Gemini API first if configured
        if gemini_client:
            try:
                if isinstance(gemini_client, OpenAI):
                    # Use Gemini API endpoint directly
                    try:
                        # Construct the API endpoint URL
                        base_url = GEMINI_API_URL.rstrip('/')
                        api_endpoint = f"{base_url}/v1beta/models/gemini-3-pro-image-preview:generateContent"
                        
                        # Prepare the request payload
                        payload = {
                            "contents": [{
                                "parts": [{
                                    "text": prompt
                                }]
                            }]
                        }
                        
                        # Make the API request
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {GEMINI_API_KEY}"
                        }
                        
                        response = requests.post(
                            api_endpoint,
                            json=payload,
                            headers=headers,
                            timeout=120
                        )
                        
                        if not response.ok:
                            raise HTTPException(
                                status_code=response.status_code,
                                detail=f"Gemini API error: {response.text}"
                            )
                        
                        result = response.json()
                        
                        # Extract image data from response
                        # Gemini API returns image in candidates[0].content.parts[0].inlineData
                        if "candidates" in result and len(result["candidates"]) > 0:
                            candidate = result["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                for part in candidate["content"]["parts"]:
                                    if "inlineData" in part:
                                        image_data = part["inlineData"]["data"]
                                        mime_type = part["inlineData"].get("mimeType", "image/png")
                                        # image_data is already base64 encoded, use it directly
                                        return f"data:{mime_type};base64,{image_data}"
                        
                        raise HTTPException(
                            status_code=500,
                            detail="No image data found in Gemini API response"
                        )
                    except HTTPException:
                        raise
                    except Exception as e:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Gemini API error: {str(e)}"
                        )
                elif gemini_client == "native" and GEMINI_AVAILABLE:
                    # Use native Google Generative AI library
                    model = genai.GenerativeModel('gemini-3-pro-image-preview')
                    response = model.generate_content(prompt)
                    
                    # Handle response based on API format
                    if hasattr(response, 'images') and response.images:
                        image_data = response.images[0]
                        if isinstance(image_data, bytes):
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                            return f"data:image/png;base64,{image_base64}"
                    
                    # Fallback: return a placeholder
                    return "https://placeholder.com/gemini-image"
            except Exception as gemini_error:
                # If Gemini fails, fall back to DALL-E 3
                print(f"Gemini API error: {gemini_error}, falling back to DALL-E 3")
        
        # Use DALL-E 3 as fallback or primary option
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        # Extract the image URL
        image_url = response.data[0].url
        return image_url
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error rendering image: {str(e)}"
        )


@app.post("/api/render", response_model=RenderImageResponse)
async def render(request: GenerateImageRequest):
    """
    POST endpoint for rendering Mermaid code into an image.
    
    Process:
    1. Call generate_visual_schema to transform Mermaid code into Visual Schema
    2. Call render_image to generate the image from the schema
    
    Args:
        request: GenerateImageRequest containing mermaid_code and style
        
    Returns:
        RenderImageResponse containing the schema and image_url
    """
    try:
        # Phase 2: Generate Visual Schema from Mermaid code
        visual_schema = generate_visual_schema(request.mermaid_code)
        
        # Phase 3: Generate Image from Visual Schema
        image_url = render_image(visual_schema, request.style)
        
        return RenderImageResponse(
            visual_schema=visual_schema,
            image_url=image_url
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Error in render pipeline: {str(e)}"
        )

