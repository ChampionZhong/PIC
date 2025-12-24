"""
Backend for Paper Illustration Copilot - Multi-Agent Workflow
"""
import os
import sys
import base64
import json
import asyncio
import requests
import google.generativeai as genai
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from prompts import (
        MANAGER_SYSTEM_PROMPT,
        ARCHITECT_SYSTEM_PROMPT,
        DIRECTOR_SYSTEM_PROMPT_TEMPLATE,
        CRITIC_SYSTEM_PROMPT,
        LOGIC_CRITIC_SYSTEM_PROMPT,
        STYLE_CRITIC_SYSTEM_PROMPT,
        RESULT_CRITIC_SYSTEM_PROMPT,
        STYLE_PRESETS
    )
except ImportError:
    # Fallback if running as module
    try:
        from backend.prompts import (
            MANAGER_SYSTEM_PROMPT,
            ARCHITECT_SYSTEM_PROMPT,
            DIRECTOR_SYSTEM_PROMPT_TEMPLATE,
            CRITIC_SYSTEM_PROMPT,
            LOGIC_CRITIC_SYSTEM_PROMPT,
            STYLE_CRITIC_SYSTEM_PROMPT,
            RESULT_CRITIC_SYSTEM_PROMPT,
            STYLE_PRESETS
        )
    except ImportError:
        # Last resort: try relative import
        from .prompts import (
            MANAGER_SYSTEM_PROMPT,
            ARCHITECT_SYSTEM_PROMPT,
            DIRECTOR_SYSTEM_PROMPT_TEMPLATE,
            CRITIC_SYSTEM_PROMPT,
            LOGIC_CRITIC_SYSTEM_PROMPT,
            STYLE_CRITIC_SYSTEM_PROMPT,
            RESULT_CRITIC_SYSTEM_PROMPT,
            STYLE_PRESETS
        )

GEMINI_AVAILABLE = True

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Research Figure Generator API - Multi-Agent Workflow")

# Enable CORS for React frontend
# Allow all origins in development, restrict in production
cors_origins_str = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8001,http://localhost:3000,http://127.0.0.1:8001,http://127.0.0.1:3000"
)
# Split and strip whitespace from each origin
cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Get OpenAI configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")

# Initialize OpenAI client (mock if API key is missing)
openai_client = None
if OPENAI_API_KEY:
    client_config = {"api_key": OPENAI_API_KEY}
    if OPENAI_API_URL:
        client_config["base_url"] = OPENAI_API_URL
    openai_client = OpenAI(**client_config)
else:
    print("Warning: OPENAI_API_KEY not found. Agent functions will use mock responses.")

# Get Gemini configuration from environment (optional)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")

# Initialize Gemini client
gemini_client = None
if GEMINI_API_KEY:
    if GEMINI_API_URL:
        gemini_client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_API_URL
        )
    elif GEMINI_AVAILABLE:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = "native"

# ============================================================================
# Task 1: Data Models for Workflow State
# ============================================================================

class LogicResponse(BaseModel):
    """Response from The Architect agent"""
    analysis: str = Field(description="Logic analysis explaining abstraction choices")
    mermaid_code: str = Field(description="Valid Mermaid JS graph code (LR direction)")


class StyleResponse(BaseModel):
    """Response from The Art Director agent"""
    visual_schema: str = Field(description="Visual schema text description for image generation")


class PaintResponse(BaseModel):
    """Response from The Painter agent"""
    image_url: str = Field(description="URL or base64 data URL of the generated image")


class CritiqueResponse(BaseModel):
    """Response from The Critic agent"""
    feedback: str = Field(description="Specific critique of the image")
    passed: bool = Field(description="Whether the image passes the review (PASS/FAIL)")
    suggestions: Optional[str] = Field(default=None, description="Specific suggestions for improvement")


class LogicCriticRequest(BaseModel):
    """Request for Logic Critic"""
    mermaid_code: str = Field(description="Mermaid code to review")
    original_idea: Optional[str] = Field(default=None, description="Original user idea for context")


class StyleCriticRequest(BaseModel):
    """Request for Style Critic"""
    visual_schema: str = Field(description="Visual schema to review")
    mermaid_code: Optional[str] = Field(default=None, description="Original Mermaid code for context")
    style_mode: Optional[str] = Field(default=None, description="Selected style mode")


class ResultCriticRequest(BaseModel):
    """Request for Result Critic"""
    image_url: str = Field(description="Image URL from The Painter")
    original_idea: str = Field(description="Original user idea for comparison")
    visual_schema: Optional[str] = Field(default=None, description="Visual schema used for generation")


# Request models for API endpoints
class ArchitectRequest(BaseModel):
    idea: str = Field(description="User's vague idea for the scientific figure")


class DirectorRequest(BaseModel):
    mermaid_code: str = Field(description="Mermaid code from The Architect")
    user_feedback: Optional[str] = Field(default=None, description="Optional user feedback for refinement")
    style_mode: str = Field(default="AI_CONFERENCE", description="Style preset: AI_CONFERENCE, TOP_JOURNAL, ENGINEERING, or CUSTOM")
    custom_style_prompt: Optional[str] = Field(default=None, description="Custom style prompt (only used if style_mode is CUSTOM)")


class PainterRequest(BaseModel):
    visual_schema: str = Field(description="Visual schema from The Art Director")


class CriticRequest(BaseModel):
    image_url: str = Field(description="Image URL from The Painter")
    original_idea: str = Field(description="Original user idea for comparison")


# Manager Agent Models
class ChatRequest(BaseModel):
    """Request for unified chat endpoint"""
    message: str = Field(description="User's message")
    history: List[dict] = Field(default=[], description="Chat history")
    active_tab: Optional[str] = Field(default=None, description="Current active tab: 'logic' | 'style' | 'result'")
    current_artifacts: Optional[dict] = Field(default_factory=dict, description="Current artifacts: {mermaid_code, visual_schema, image_url}")
    style_mode: Optional[str] = Field(default=None, description="Style mode for Art Director")
    custom_style_prompt: Optional[str] = Field(default=None, description="Custom style prompt for Art Director")
    style_mode: Optional[str] = Field(default="AI_CONFERENCE", description="Style preset: AI_CONFERENCE, TOP_JOURNAL, ENGINEERING, or CUSTOM")
    custom_style_prompt: Optional[str] = Field(default=None, description="Custom style prompt (only used if style_mode is CUSTOM)")


class ChatResponse(BaseModel):
    """Unified response from Manager Agent"""
    manager_reasoning: str = Field(description="Manager's reasoning for routing")
    agent_name: str = Field(description="Target agent name: 'architect' | 'art_director' | 'painter' | 'critic'")
    response_text: str = Field(description="Agent's response text")
    updated_artifacts: dict = Field(default_factory=dict, description="Updated artifacts (only what changed)")


# ============================================================================
# Task 2: Agent Functions
# ============================================================================


async def run_manager(user_input: str, context: dict) -> dict:
    """
    Manager Agent: Classifies user intent and routes to the correct specialist agent.
    
    Args:
        user_input: User's message
        context: Context dictionary containing active_tab and current_artifacts
        
    Returns:
        Dictionary with target_agent and reasoning
    """
    if not openai_client:
        # Mock response if API key is missing
        return {
            "target_agent": "architect",
            "reasoning": "Mock routing: Defaulting to architect. Please configure OPENAI_API_KEY."
        }
    
    try:
        active_tab = context.get("active_tab", "logic")
        current_artifacts = context.get("current_artifacts", {})
        
        context_info = f"""
Current System State:
- Active Tab: {active_tab}
- Available Artifacts: {', '.join(current_artifacts.keys()) if current_artifacts else 'None'}
"""
        
        messages = [
            {"role": "system", "content": MANAGER_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context_info}\n\nUser Message: {user_input}\n\nDetermine which agent should handle this request and return JSON."}
        ]
        
        # Run synchronous OpenAI call in thread pool to avoid blocking
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-5.2",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON response
        try:
            result = json.loads(content)
            target_agent = result.get("target_agent", "architect")
            reasoning = result.get("reasoning", "No reasoning provided")
            
            return {
                "target_agent": target_agent,
                "reasoning": reasoning
            }
        except json.JSONDecodeError:
            # Fallback: try to extract from text
            if "architect" in content.lower():
                return {"target_agent": "architect", "reasoning": content[:200]}
            elif "art_director" in content.lower() or "director" in content.lower():
                return {"target_agent": "art_director", "reasoning": content[:200]}
            elif "painter" in content.lower():
                return {"target_agent": "painter", "reasoning": content[:200]}
            elif "critic" in content.lower():
                return {"target_agent": "critic", "reasoning": content[:200]}
            else:
                return {"target_agent": "architect", "reasoning": "Default routing"}
    
    except Exception as e:
        print(f"Manager routing error: {e}")
        # Default to architect on error
        return {
            "target_agent": "architect",
            "reasoning": f"Error in routing, defaulting to architect: {str(e)}"
        }


async def run_architect(idea: str) -> LogicResponse:
    """
    The Architect Agent: Converts vague user ideas -> Mermaid Code (Logic).
    
    Args:
        idea: User's vague idea for the scientific figure
        
    Returns:
        LogicResponse containing analysis and mermaid_code
        
    Raises:
        HTTPException: If the API call fails
    """
    if not openai_client:
        # Mock response if API key is missing
        return LogicResponse(
            analysis="Mock analysis: This is a placeholder response. Please configure OPENAI_API_KEY.",
            mermaid_code="graph LR\n\nA[Input] --> B[Process]\nB --> C[Output]"
        )
    
    try:
        messages = [
            {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this idea and generate a Mermaid diagram:\n\n{idea}"}
        ]
        
        # Run synchronous OpenAI call in thread pool to avoid blocking
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-5.2",  # Can be overridden via env
            messages=messages,
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        
        # Extract analysis and mermaid code
        analysis = ""
        mermaid_code = ""
        
        if "[[ANALYSIS_START]]" in content and "[[ANALYSIS_END]]" in content:
            start_idx = content.find("[[ANALYSIS_START]]") + len("[[ANALYSIS_START]]")
            end_idx = content.find("[[ANALYSIS_END]]")
            analysis = content[start_idx:end_idx].strip()
        
        if "[[MERMAID_START]]" in content and "[[MERMAID_END]]" in content:
            start_idx = content.find("[[MERMAID_START]]") + len("[[MERMAID_START]]")
            end_idx = content.find("[[MERMAID_END]]")
            mermaid_code = content[start_idx:end_idx].strip()
        else:
            # Fallback: try to extract mermaid code blocks
            if "```mermaid" in content:
                start_idx = content.find("```mermaid") + len("```mermaid")
                end_idx = content.find("```", start_idx)
                mermaid_code = content[start_idx:end_idx].strip()
            elif "```" in content:
                # Generic code block
                parts = content.split("```")
                if len(parts) >= 3:
                    mermaid_code = parts[1].strip()
                    if mermaid_code.startswith("mermaid"):
                        mermaid_code = mermaid_code[7:].strip()
        
        if not analysis:
            analysis = "Analysis: Generated Mermaid diagram based on the provided idea."
        if not mermaid_code:
            mermaid_code = content  # Fallback to full content
        
        return LogicResponse(analysis=analysis, mermaid_code=mermaid_code)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Architect agent: {str(e)}"
        )


async def run_art_director(
    mermaid_code: str, 
    user_feedback: Optional[str] = None,
    style_mode: str = "AI_CONFERENCE",
    custom_style_prompt: Optional[str] = None
) -> StyleResponse:
    """
    The Art Director Agent: Converts Mermaid Code -> Visual Schema (Style Description).
    
    Args:
        mermaid_code: Mermaid code from The Architect
        user_feedback: Optional user feedback for refinement
        style_mode: Style preset mode (AI_CONFERENCE, TOP_JOURNAL, ENGINEERING, CUSTOM)
        custom_style_prompt: Custom style prompt (only used if style_mode is CUSTOM)
        
    Returns:
        StyleResponse containing visual_schema
        
    Raises:
        HTTPException: If the API call fails
    """
    if not openai_client:
        # Mock response if API key is missing
        return StyleResponse(
            visual_schema="[Style & Meta-Instructions]\nHigh-fidelity scientific schematic, clean white background.\n\n[LAYOUT CONFIGURATION]\n* Selected Layout: Linear\n* Composition Logic: Left to right flow\n*"
        )
    
    try:
        # Select style injection based on style_mode
        if style_mode not in STYLE_PRESETS:
            style_mode = "AI_CONFERENCE"  # Fallback to default
        
        style_config = STYLE_PRESETS[style_mode]
        
        # Handle custom style
        if style_mode == "CUSTOM":
            if custom_style_prompt:
                style_injection = f"""
[TARGET STYLE: Custom User Style]
- **User Requirement:** "{custom_style_prompt}"
- **Instruction:** Interpret this style artistically but maintain scientific clarity.
"""
            else:
                # Fallback to default if CUSTOM but no prompt provided
                style_injection = STYLE_PRESETS["AI_CONFERENCE"]["prompt_injection"]
        else:
            style_injection = style_config["prompt_injection"]
        
        # Build system prompt with dynamic style injection
        director_system_prompt = DIRECTOR_SYSTEM_PROMPT_TEMPLATE.format(
            style_injection_block=style_injection.strip()
        )
        
        user_content = f"Translate this Mermaid code into a Visual Schema:\n\n{mermaid_code}"
        if user_feedback:
            user_content += f"\n\nUser Feedback: {user_feedback}"
        
        messages = [
            {"role": "system", "content": director_system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Run synchronous OpenAI call in thread pool to avoid blocking
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-5.2",
            messages=messages,
            temperature=0.7,
        )
        
        visual_schema = response.choices[0].message.content.strip()
        
        # Extract schema from markers if present
        if "---BEGIN PROMPT---" in visual_schema and "---END PROMPT---" in visual_schema:
            start_idx = visual_schema.find("---BEGIN PROMPT---") + len("---BEGIN PROMPT---")
            end_idx = visual_schema.find("---END PROMPT---")
            visual_schema = visual_schema[start_idx:end_idx].strip()
        
        return StyleResponse(visual_schema=visual_schema)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Art Director agent: {str(e)}"
        )


async def run_painter(visual_schema: str) -> PaintResponse:
    """
    The Painter Agent: Converts Visual Schema -> Image (using Gemini).
    
    Args:
        visual_schema: Visual schema from The Art Director
        
    Returns:
        PaintResponse containing image_url
        
    Raises:
        HTTPException: If the image generation fails
    """
    # Build the prompt with style instructions
    prompt_template = """**Style Reference & Execution Instructions:**

    1. **Art Style (Visio/Illustrator Aesthetic):**
        Generate a **professional academic architecture diagram** suitable for a top-tier computer science paper (CVPR/NeurIPS).
        * **Visuals:** Flat vector graphics, distinct geometric shapes, clean thin outlines, and soft pastel fills (Azure Blue, Slate Grey, Coral Orange).
        * **Layout:** Strictly follow the spatial arrangement defined below.
        * **Vibe:** Technical, precise, clean white background. NOT hand-drawn, NOT photorealistic, NOT 3D render, NO shadows/shading.

    2. **CRITICAL TEXT CONSTRAINTS (Read Carefully):**
        * **DO NOT render meta-labels:** Do not write words like "ZONE 1", "LAYOUT CONFIGURATION", "Input", "Output", or "Container" inside the image. These are structural instructions for YOU, not text for the image.
        * **ONLY render "Key Text Labels":** Only text inside double quotes (e.g., "[Text]") listed under "Key Text Labels" should appear in the diagram.
        * **Font:** Use a clean, bold Sans-Serif font (like Roboto or Helvetica) for all labels.

    3. **Visual Schema Execution:**
        Translate the following structural blueprint into the final image:

    {visual_schema}"""
    
    # Truncate visual_schema if too long
    wrapper_length = len(prompt_template.replace('{visual_schema}', ''))
    MAX_TOTAL_LENGTH = 4000
    MAX_SCHEMA_LENGTH = MAX_TOTAL_LENGTH - wrapper_length - 100
    
    if len(visual_schema) > MAX_SCHEMA_LENGTH:
        truncated = visual_schema[:MAX_SCHEMA_LENGTH]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        cutoff_point = max(last_period, last_newline) if max(last_period, last_newline) > MAX_SCHEMA_LENGTH * 0.8 else MAX_SCHEMA_LENGTH
        if cutoff_point > MAX_SCHEMA_LENGTH * 0.8:
            visual_schema = truncated[:cutoff_point + 1] + "\n[Schema truncated for length]"
        else:
            visual_schema = truncated + "... [Schema truncated for length]"
    
    prompt = prompt_template.format(visual_schema=visual_schema)
    
    if len(prompt) > MAX_TOTAL_LENGTH:
        prompt = prompt[:MAX_TOTAL_LENGTH - 50] + "... [Prompt truncated]"
    
    # Check if Gemini API is configured
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is required for image generation. Please configure GEMINI_API_KEY."
        )
    
    try:
        # Try Gemini API HTTP endpoint first (if GEMINI_API_URL is configured)
        http_api_failed = False
        http_api_error = None
        
        if GEMINI_API_URL:
            try:
                # Use Gemini API with generationConfig using requests
                from urllib.parse import urlparse
                
                api_url = GEMINI_API_URL.rstrip('/')
                
                # Build endpoint
                default_endpoint = "/v1beta/models/gemini-3-pro-image-preview:generateContent/"
                
                # Parse custom API URL
                parsed = urlparse(api_url)
                
                # Handle path: if it's a base URL, append the model endpoint; if it's full path, use as-is
                if parsed.path:
                    base_path = parsed.path.rstrip('/')
                    # If the path already contains the full endpoint, use it as-is
                    if "gemini-3-pro-image-preview:generateContent" in base_path:
                        endpoint = base_path + ("/" if not base_path.endswith("/") else "")
                    else:
                        # Otherwise, treat it as a base path and append the model endpoint
                        # If base_path ends with /v1beta, append /models/...
                        if base_path.endswith("/v1beta"):
                            endpoint = base_path + "/models/gemini-3-pro-image-preview:generateContent/"
                        # If base_path is empty or just "/", use default endpoint
                        elif base_path == "" or base_path == "/":
                            endpoint = default_endpoint
                        # Otherwise, append /v1beta/models/... to the base path
                        else:
                            endpoint = base_path.rstrip('/') + default_endpoint
                else:
                    endpoint = default_endpoint
                
                # Build full URL
                url = api_url + endpoint
                
                payload = {
                    "contents": [{
                        "role": "user",
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "responseModalities": [
                            "IMAGE"  # Request only IMAGE, not TEXT, to avoid getting thoughts
                        ],
                        "imageConfig": {
                            "aspectRatio": "16:9",
                            "imageSize": "4K"
                        }
                    }
                }
                
                headers = {
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                # Use requests.request in thread pool to avoid blocking
                response = await asyncio.to_thread(
                    requests.request,
                    "POST",
                    url,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=120
                )
                
                # Check response status
                if response.status_code != 200:
                    raise Exception(
                        f"Gemini API request failed with status {response.status_code}\n"
                        f"Response text: {response.text[:500]}"
                    )
                
                # Check if response is HTML (common error case)
                response_text = response.text.strip()
                if response_text.startswith("<!doctype html>") or response_text.startswith("<html"):
                    raise Exception(
                        f"Received HTML response instead of JSON. This usually means the API endpoint is incorrect.\n"
                        f"URL: {url}\n"
                        f"Response preview: {response_text[:500]}..."
                    )
                
                # Check if response is empty
                if not response_text:
                    raise Exception(
                        f"Received empty response from API.\n"
                        f"URL: {url}\n"
                        f"Status code: {response.status_code}"
                    )
                
                # Try to parse JSON
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    raise Exception(
                        f"Failed to parse JSON response.\n"
                        f"Error: {e}\n"
                        f"Response text (first 500 chars): {response_text[:500]}...\n"
                        f"Full response length: {len(response_text)}"
                    )
                
                # Parse response
                candidates = result.get("candidates", [])
                
                if not candidates:
                    raise Exception(
                        f"No candidates found in API response.\n"
                        f"Response JSON: {json.dumps(result, indent=2)[:1000]}..."
                    )
                
                parts = candidates[0].get("content", {}).get("parts", [])
                
                if not parts:
                    raise Exception(
                        f"No parts found in candidate content.\n"
                        f"Candidate: {json.dumps(candidates[0], indent=2)[:1000]}..."
                    )
                
                # Extract image data
                image_data_b64 = None
                mime_type = "image/png"
                text_parts = []
                thought_parts = []
                
                for part in parts:
                    if "inlineData" in part:
                        image_data_b64 = part["inlineData"]["data"]
                        mime_type = part["inlineData"].get("mimeType", "image/png")
                        break
                    elif "text" in part:
                        text_content = part.get("text", "")
                        if part.get("thought", False):
                            thought_parts.append(text_content)
                        else:
                            text_parts.append(text_content)
                
                if image_data_b64:
                    image_url = f"data:{mime_type};base64,{image_data_b64}"
                    return PaintResponse(image_url=image_url)
                else:
                    # Check if we got thoughts but no image - this might be a streaming response
                    if thought_parts:
                        error_msg = (
                            f"Gemini API returned thinking process but no image yet. "
                            f"This might be a streaming response that needs more time.\n\n"
                            f"Thoughts received:\n" + "\n".join(thought_parts[:3]) + "\n\n"
                            f"Total parts: {len(parts)}, Text parts: {len(text_parts)}, Thought parts: {len(thought_parts)}\n\n"
                            f"Please try again or check if the API supports streaming responses."
                        )
                    elif text_parts:
                        error_msg = (
                            f"Gemini API returned text response instead of image.\n\n"
                            f"Text received:\n" + "\n".join(text_parts[:2]) + "\n\n"
                            f"This might indicate the API is still processing or the prompt needs adjustment."
                        )
                    else:
                        error_msg = (
                            f"No image data found in response parts.\n"
                            f"Parts structure: {json.dumps(parts[:2], indent=2) if parts else 'No parts'}..."
                        )
                    
                    raise Exception(error_msg)
                    
            except Exception as gemini_error:
                http_api_failed = True
                http_api_error = gemini_error
                # Will try native library below if available
        
        # Try native Google Generative AI library (if HTTP API failed or not configured)
        if (http_api_failed or not GEMINI_API_URL) and GEMINI_AVAILABLE and gemini_client == "native":
            try:
                # Run synchronous Gemini call in thread pool to avoid blocking
                def generate_gemini_content():
                    model = genai.GenerativeModel('gemini-3-pro-image-preview')
                    return model.generate_content(prompt)
                
                response = await asyncio.to_thread(generate_gemini_content)
                
                if hasattr(response, 'images') and response.images:
                    image_data = response.images[0]
                    if isinstance(image_data, bytes):
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        return PaintResponse(image_url=f"data:image/png;base64,{image_base64}")
                else:
                    raise Exception("No images found in Gemini response")
            except Exception as native_error:
                # If both HTTP and native failed, raise error with details
                error_msg = "Gemini image generation failed."
                if http_api_failed:
                    error_msg += f" HTTP API error: {str(http_api_error)}"
                error_msg += f" Native API error: {str(native_error)}"
                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )
        
        # If we reach here, no Gemini method worked
        if http_api_failed:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini HTTP API failed: {str(http_api_error)}. Native library not available."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Gemini image generation failed. Please check GEMINI_API_KEY and GEMINI_API_URL configuration."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Painter agent: {str(e)}"
        )


async def run_critic(image_url: str, original_idea: str) -> CritiqueResponse:
    """
    The Critic Agent: Reviews the image and provides feedback.
    
    Args:
        image_url: Image URL from The Painter (can be HTTP URL or data URL)
        original_idea: Original user idea for comparison
        
    Returns:
        CritiqueResponse containing feedback and passed status
        
    Raises:
        HTTPException: If the API call fails
    """
    if not openai_client:
        # Mock response if API key is missing
        return CritiqueResponse(
            feedback="Mock critique: This is a placeholder response. Please configure OPENAI_API_KEY.",
            passed=True
        )
    
    try:
        # Use vision model for image review
        system_prompt = CRITIC_SYSTEM_PROMPT

        user_content = f"""Review this image against the original idea:

            Original Idea: {original_idea}

            Check:
            1. Does the image match the original idea?
            2. Is all text legible?
            3. Are there any visual issues or inconsistencies?
            4. Would this be suitable for a top-tier conference paper?"""
        
        # Handle data URL vs HTTP URL
        # OpenAI Vision API supports data URLs in the format: data:image/<type>;base64,<base64_data>
        if image_url.startswith('data:'):
            # Extract and validate base64 data from data URL
            # Format: data:image/png;base64,<base64_data>
            try:
                # Parse the data URL
                if ',' not in image_url:
                    raise ValueError("Invalid data URL format: missing comma separator")
                
                header, encoded = image_url.split(',', 1)
                
                # Extract mime type (e.g., image/png, image/jpeg)
                mime_type = 'image/png'  # default
                if 'image/' in header:
                    mime_type_part = header.split('image/')[1]
                    if ';' in mime_type_part:
                        mime_type = mime_type_part.split(';')[0]
                    else:
                        mime_type = mime_type_part
                
                # Validate base64 encoding
                try:
                    # Try to decode to verify it's valid base64
                    base64.b64decode(encoded, validate=True)
                except Exception as decode_error:
                    raise ValueError(f"Invalid base64 encoding: {decode_error}")
                
                # Reconstruct data URL with proper format
                # OpenAI Vision API expects: data:image/<type>;base64,<base64_string>
                formatted_url = f"data:image/{mime_type};base64,{encoded}"
                
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": formatted_url}
                }
                print(f"Processed data URL with mime type: image/{mime_type}")
            except Exception as e:
                # Fallback: try to use the URL as-is
                print(f"Error parsing data URL: {e}, using as-is")
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
        else:
            # HTTP/HTTPS URL
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    image_content
                ]
            }
        ]
        
        # Try vision-capable models (gpt-4o, gpt-4-vision-preview)
        # Note: gpt-5.2 may not support vision, so we'll try it last
        vision_models = ["gpt-4o", "gpt-4-vision-preview"]
        response = None
        last_error = None
        
        for model in vision_models:
            try:
                # Run synchronous OpenAI call in thread pool to avoid blocking
                response = await asyncio.to_thread(
                    openai_client.chat.completions.create,
                    model=model,
                    messages=messages,
                    temperature=0.5,
                )
                print(f"Successfully used vision model: {model}")
                break
            except Exception as e:
                last_error = e
                print(f"Vision model {model} failed: {e}, trying next model...")
                continue
        
        # If all vision models failed, fallback to text-only
        if response is None:
            print(f"All vision models failed, using text-only critique. Last error: {last_error}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Review this image (URL: {image_url[:100] if len(image_url) > 100 else image_url}...) against the original idea: {original_idea}. Note: Image review is limited without vision capability."}
            ]
            # Run synchronous OpenAI call in thread pool to avoid blocking
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-5.2",
                messages=messages,
                temperature=0.5,
            )
        
        content = response.choices[0].message.content
        
        # Extract pass/fail and feedback
        passed = "[[PASS]]" in content or "PASS" in content.upper()
        feedback = ""
        
        if "[[FEEDBACK_START]]" in content and "[[FEEDBACK_END]]" in content:
            start_idx = content.find("[[FEEDBACK_START]]") + len("[[FEEDBACK_START]]")
            end_idx = content.find("[[FEEDBACK_END]]")
            feedback = content[start_idx:end_idx].strip()
        else:
            feedback = content.strip()
        
        if not feedback:
            feedback = "Review completed."
        
        return CritiqueResponse(feedback=feedback, passed=passed)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Critic agent: {str(e)}"
        )


async def run_logic_critic(mermaid_code: str, original_idea: Optional[str] = None) -> CritiqueResponse:
    """
    Logic Critic Agent: Reviews Mermaid code quality, structure, and logic.
    
    Args:
        mermaid_code: Mermaid code to review
        original_idea: Original user idea for context
        
    Returns:
        CritiqueResponse containing feedback, passed status, and suggestions
    """
    if not openai_client:
        return CritiqueResponse(
            feedback="Mock critique: Please configure OPENAI_API_KEY.",
            passed=True,
            suggestions="Configure API key to get real critique."
        )
    
    try:
        system_prompt = LOGIC_CRITIC_SYSTEM_PROMPT

        user_content = f"Review this Mermaid code:\n\n```mermaid\n{mermaid_code}\n```\n"
        if original_idea:
            user_content += f"\nOriginal idea: {original_idea}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Run synchronous OpenAI call in thread pool to avoid blocking
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-5.2",
            messages=messages,
            temperature=0.5,
        )
        
        content = response.choices[0].message.content
        
        # Extract pass/fail, feedback, and suggestions
        passed = "[[PASS]]" in content or "PASS" in content.upper()
        feedback = ""
        suggestions = ""
        
        if "[[FEEDBACK_START]]" in content and "[[FEEDBACK_END]]" in content:
            start_idx = content.find("[[FEEDBACK_START]]") + len("[[FEEDBACK_START]]")
            end_idx = content.find("[[FEEDBACK_END]]")
            feedback = content[start_idx:end_idx].strip()
        else:
            feedback = content.strip()
        
        if "[[SUGGESTIONS_START]]" in content and "[[SUGGESTIONS_END]]" in content:
            start_idx = content.find("[[SUGGESTIONS_START]]") + len("[[SUGGESTIONS_START]]")
            end_idx = content.find("[[SUGGESTIONS_END]]")
            suggestions = content[start_idx:end_idx].strip()
        
        if not feedback:
            feedback = "Review completed."
        
        return CritiqueResponse(feedback=feedback, passed=passed, suggestions=suggestions)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Logic Critic agent: {str(e)}"
        )


async def run_style_critic(visual_schema: str, mermaid_code: Optional[str] = None, style_mode: Optional[str] = None) -> CritiqueResponse:
    """
    Style Critic Agent: Reviews Visual Schema quality, style consistency, and clarity.
    
    Args:
        visual_schema: Visual schema to review
        mermaid_code: Original Mermaid code for context
        style_mode: Selected style mode for context
        
    Returns:
        CritiqueResponse containing feedback, passed status, and suggestions
    """
    if not openai_client:
        return CritiqueResponse(
            feedback="Mock critique: Please configure OPENAI_API_KEY.",
            passed=True,
            suggestions="Configure API key to get real critique."
        )
    
    try:
        system_prompt = STYLE_CRITIC_SYSTEM_PROMPT

        user_content = f"Review this Visual Schema:\n\n{visual_schema}\n"
        if mermaid_code:
            user_content += f"\nOriginal Mermaid code:\n```mermaid\n{mermaid_code}\n```\n"
        if style_mode:
            user_content += f"\nSelected style mode: {style_mode}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Run synchronous OpenAI call in thread pool to avoid blocking
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-5.2",
            messages=messages,
            temperature=0.5,
        )
        
        content = response.choices[0].message.content
        
        # Extract pass/fail, feedback, and suggestions
        passed = "[[PASS]]" in content or "PASS" in content.upper()
        feedback = ""
        suggestions = ""
        
        if "[[FEEDBACK_START]]" in content and "[[FEEDBACK_END]]" in content:
            start_idx = content.find("[[FEEDBACK_START]]") + len("[[FEEDBACK_START]]")
            end_idx = content.find("[[FEEDBACK_END]]")
            feedback = content[start_idx:end_idx].strip()
        else:
            feedback = content.strip()
        
        if "[[SUGGESTIONS_START]]" in content and "[[SUGGESTIONS_END]]" in content:
            start_idx = content.find("[[SUGGESTIONS_START]]") + len("[[SUGGESTIONS_START]]")
            end_idx = content.find("[[SUGGESTIONS_END]]")
            suggestions = content[start_idx:end_idx].strip()
        
        if not feedback:
            feedback = "Review completed."
        
        return CritiqueResponse(feedback=feedback, passed=passed, suggestions=suggestions)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Style Critic agent: {str(e)}"
        )


async def run_result_critic(image_url: str, original_idea: str, visual_schema: Optional[str] = None) -> CritiqueResponse:
    """
    Result Critic Agent: Reviews the generated image quality and match to requirements.
    This is an improved version of run_critic with suggestions support.
    
    Args:
        image_url: Image URL from The Painter
        original_idea: Original user idea for comparison
        visual_schema: Visual schema used for generation
        
    Returns:
        CritiqueResponse containing feedback, passed status, and suggestions
    """
    if not openai_client:
        return CritiqueResponse(
            feedback="Mock critique: Please configure OPENAI_API_KEY.",
            passed=True,
            suggestions="Configure API key to get real critique."
        )
    
    try:
        system_prompt = RESULT_CRITIC_SYSTEM_PROMPT

        user_content = f"""Review this image against the original idea:

            Original Idea: {original_idea}

            Check:
            1. Does the image match the original idea?
            2. Is all text legible?
            3. Are there any visual issues or inconsistencies?
            4. Would this be suitable for a top-tier conference paper?"""
        
        if visual_schema:
            user_content += f"\n\nVisual Schema used:\n{visual_schema[:500]}..."  # Truncate if too long
        
        # Handle data URL vs HTTP URL (same logic as before)
        if image_url.startswith('data:'):
            try:
                if ',' not in image_url:
                    raise ValueError("Invalid data URL format: missing comma separator")
                
                header, encoded = image_url.split(',', 1)
                mime_type = 'image/png'
                if 'image/' in header:
                    mime_type_part = header.split('image/')[1]
                    if ';' in mime_type_part:
                        mime_type = mime_type_part.split(';')[0]
                    else:
                        mime_type = mime_type_part
                
                base64.b64decode(encoded, validate=True)
                formatted_url = f"data:image/{mime_type};base64,{encoded}"
                
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": formatted_url}
                }
                print(f"Processed data URL with mime type: image/{mime_type}")
            except Exception as e:
                print(f"Error parsing data URL: {e}, using as-is")
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
        else:
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    image_content
                ]
            }
        ]
        
        # Try vision-capable models
        vision_models = ["gpt-4o", "gpt-4-vision-preview"]
        response = None
        last_error = None
        
        for model in vision_models:
            try:
                # Run synchronous OpenAI call in thread pool to avoid blocking
                response = await asyncio.to_thread(
                    openai_client.chat.completions.create,
                    model=model,
                    messages=messages,
                    temperature=0.5,
                )
                print(f"Successfully used vision model: {model}")
                break
            except Exception as e:
                last_error = e
                print(f"Vision model {model} failed: {e}, trying next model...")
                continue
        
        # If all vision models failed, fallback to text-only
        if response is None:
            print(f"All vision models failed, using text-only critique. Last error: {last_error}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Review this image (URL: {image_url[:100] if len(image_url) > 100 else image_url}...) against the original idea: {original_idea}. Note: Image review is limited without vision capability."}
            ]
            # Run synchronous OpenAI call in thread pool to avoid blocking
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-5.2",
                messages=messages,
                temperature=0.5,
            )
        
        content = response.choices[0].message.content
        
        # Extract pass/fail, feedback, and suggestions
        passed = "[[PASS]]" in content or "PASS" in content.upper()
        feedback = ""
        suggestions = ""
        
        if "[[FEEDBACK_START]]" in content and "[[FEEDBACK_END]]" in content:
            start_idx = content.find("[[FEEDBACK_START]]") + len("[[FEEDBACK_START]]")
            end_idx = content.find("[[FEEDBACK_END]]")
            feedback = content[start_idx:end_idx].strip()
        else:
            feedback = content.strip()
        
        if "[[SUGGESTIONS_START]]" in content and "[[SUGGESTIONS_END]]" in content:
            start_idx = content.find("[[SUGGESTIONS_START]]") + len("[[SUGGESTIONS_START]]")
            end_idx = content.find("[[SUGGESTIONS_END]]")
            suggestions = content[start_idx:end_idx].strip()
        
        if not feedback:
            feedback = "Review completed."
        
        return CritiqueResponse(feedback=feedback, passed=passed, suggestions=suggestions)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in Result Critic agent: {str(e)}"
        )


# ============================================================================
# Task 3: API Endpoints
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Unified chat endpoint that routes user messages to the appropriate agent.
    
    Args:
        request: ChatRequest containing message, history, active_tab, and current_artifacts
        
    Returns:
        ChatResponse with manager reasoning, agent name, response text, and updated artifacts
    """
    try:
        # Build context for manager
        context = {
            "active_tab": request.active_tab or "logic",
            "current_artifacts": request.current_artifacts or {}
        }
        
        # Call manager to determine target agent
        manager_result = await run_manager(request.message, context)
        target_agent = manager_result["target_agent"]
        manager_reasoning = manager_result["reasoning"]
        
        # Dispatch to appropriate agent based on routing decision
        updated_artifacts = {}
        response_text = ""
        
        if target_agent == "architect":
            # Architect: needs user idea
            result = await run_architect(request.message)
            response_text = result.analysis
            updated_artifacts["mermaid_code"] = result.mermaid_code
            
        elif target_agent == "art_director":
            # Art Director: needs mermaid_code from artifacts or user message
            mermaid_code = request.current_artifacts.get("mermaid_code", "")
            if not mermaid_code:
                # If no mermaid code, try to extract from user message or use a default
                response_text = "Please provide Mermaid code first, or generate a diagram using the Architect."
                updated_artifacts = {}
            else:
                # Extract style parameters from request
                style_mode = request.style_mode or "AI_CONFERENCE"
                custom_style_prompt = request.custom_style_prompt
                result = await run_art_director(mermaid_code, request.message, style_mode, custom_style_prompt)
                response_text = f"I've updated the visual schema based on your feedback: {request.message}"
                updated_artifacts["visual_schema"] = result.visual_schema
                
        elif target_agent == "painter":
            # Painter: needs visual_schema from artifacts
            visual_schema = request.current_artifacts.get("visual_schema", "")
            if not visual_schema:
                # If no visual schema but we have mermaid_code, route to art_director instead
                mermaid_code = request.current_artifacts.get("mermaid_code", "")
                if mermaid_code:
                    # Force route to art_director
                    style_mode = request.style_mode or "AI_CONFERENCE"
                    custom_style_prompt = request.custom_style_prompt
                    result = await run_art_director(mermaid_code, request.message, style_mode, custom_style_prompt)
                    response_text = f"I've generated the visual schema from the Mermaid diagram."
                    updated_artifacts["visual_schema"] = result.visual_schema
                else:
                    response_text = "Please generate a visual schema first using the Art Director."
                    updated_artifacts = {}
            else:
                result = await run_painter(visual_schema)
                response_text = "Image generated successfully!"
                updated_artifacts["image_url"] = result.image_url
                
        elif target_agent == "critic":
            # Critic: needs image_url and original_idea
            image_url = request.current_artifacts.get("image_url", "")
            original_idea = request.message  # Use current message as idea, or could extract from history
            
            # Try to find original idea from history
            if request.history:
                for msg in reversed(request.history):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        original_idea = msg.get("content", request.message)
                        break
            
            if not image_url:
                response_text = "Please generate an image first using the Painter."
                updated_artifacts = {}
            else:
                result = await run_critic(image_url, original_idea)
                status = "✓ PASSED" if result.passed else "✗ FAILED"
                response_text = f"{status}\n\n{result.feedback}"
                updated_artifacts["critique"] = result.feedback
        else:
            # Fallback to architect
            result = await run_architect(request.message)
            response_text = result.analysis
            updated_artifacts["mermaid_code"] = result.mermaid_code
        
        return ChatResponse(
            manager_reasoning=manager_reasoning,
            agent_name=target_agent,
            response_text=response_text,
            updated_artifacts=updated_artifacts
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in chat endpoint: {str(e)}"
        )


@app.post("/api/agent/architect", response_model=LogicResponse)
async def architect_endpoint(request: ArchitectRequest):
    """
    POST endpoint for The Architect agent.
    Converts vague user ideas -> Mermaid Code (Logic).
    
    Args:
        request: ArchitectRequest containing user idea
        
    Returns:
        LogicResponse containing analysis and mermaid_code
    """
    return await run_architect(request.idea)


@app.post("/api/agent/director", response_model=StyleResponse)
async def director_endpoint(request: DirectorRequest):
    """
    POST endpoint for The Art Director agent.
    Converts Mermaid Code -> Visual Schema (Style Description).
    
    Args:
        request: DirectorRequest containing mermaid_code, optional user_feedback, style_mode, and custom_style_prompt
        
    Returns:
        StyleResponse containing visual_schema
    """
    return await run_art_director(
        request.mermaid_code, 
        request.user_feedback,
        request.style_mode,
        request.custom_style_prompt
    )


@app.post("/api/agent/painter", response_model=PaintResponse)
async def painter_endpoint(request: PainterRequest):
    """
    POST endpoint for The Painter agent.
    Converts Visual Schema -> Image (using Gemini).
    
    Args:
        request: PainterRequest containing visual_schema
        
    Returns:
        PaintResponse containing image_url
    """
    return await run_painter(request.visual_schema)


@app.post("/api/agent/critic", response_model=CritiqueResponse)
async def critic_endpoint(request: CriticRequest):
    """
    POST endpoint for The Critic agent.
    Reviews the image and provides feedback.
    
    Args:
        request: CriticRequest containing image_url and original_idea
        
    Returns:
        CritiqueResponse containing feedback and passed status
    """
    return await run_critic(request.image_url, request.original_idea)


@app.post("/api/agent/critic/logic", response_model=CritiqueResponse)
async def logic_critic_endpoint(request: LogicCriticRequest):
    """
    POST endpoint for Logic Critic agent.
    Reviews Mermaid code quality and provides feedback with suggestions.
    
    Args:
        request: LogicCriticRequest containing mermaid_code and optional original_idea
        
    Returns:
        CritiqueResponse containing feedback, passed status, and suggestions
    """
    return await run_logic_critic(request.mermaid_code, request.original_idea)


@app.post("/api/agent/critic/style", response_model=CritiqueResponse)
async def style_critic_endpoint(request: StyleCriticRequest):
    """
    POST endpoint for Style Critic agent.
    Reviews Visual Schema quality and provides feedback with suggestions.
    
    Args:
        request: StyleCriticRequest containing visual_schema and optional context
        
    Returns:
        CritiqueResponse containing feedback, passed status, and suggestions
    """
    return await run_style_critic(request.visual_schema, request.mermaid_code, request.style_mode)


@app.post("/api/agent/critic/result", response_model=CritiqueResponse)
async def result_critic_endpoint(request: ResultCriticRequest):
    """
    POST endpoint for Result Critic agent.
    Reviews the generated image and provides feedback with suggestions.
    
    Args:
        request: ResultCriticRequest containing image_url, original_idea, and optional visual_schema
        
    Returns:
        CritiqueResponse containing feedback, passed status, and suggestions
    """
    return await run_result_critic(request.image_url, request.original_idea, request.visual_schema)


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "AI Research Figure Generator API - Multi-Agent Workflow is running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


# ============================================================================
# Figure to PPT Pipeline API
# ============================================================================

@app.post("/api/figure-to-ppt")
async def figure_to_ppt_endpoint(
    fig_draft_file: UploadFile = File(..., description="Draft image file with content"),
    mask_detail_level: int = Form(3, description="MinerU recursion depth"),
    figure_complex: str = Form("medium", description="Figure complexity: easy/medium/hard"),
    mineru_port: Optional[int] = Form(None, description="MinerU service port (optional, only for local service mode)"),
):
    """
    Convert figure images to PPT presentation.
    
    This endpoint:
    - Receives draft image file (with content)
    - Automatically generates layout template image
    - Processes images through pipeline:
      1. Generate layout template from draft image
      2. SAM segmentation of layout elements
      3. MinerU parsing of content elements
      4. Background removal for icons
      5. PPT generation
    
    Returns the generated PPT file directly.
    """
    import tempfile
    import shutil
    import sys
    from pathlib import Path
    
    # Add project root to path if not already there
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from backend.figure_to_ppt_service import run_figure_to_ppt_pipeline_with_auto_layout
    from utils.logger import get_logger
    
    log = get_logger(__name__)
    
    # Create temporary directory for processing
    temp_dir = Path(tempfile.mkdtemp(prefix="figure_to_ppt_"))
    
    try:
        log.info(f"[figure_to_ppt_endpoint] Starting pipeline")
        log.info(f"  Uploaded file: {fig_draft_file.filename}")
        log.info(f"  MinerU port: {mineru_port}")
        log.info(f"  Mask detail level: {mask_detail_level}")
        log.info(f"  Figure complexity: {figure_complex}")
        
        # Save uploaded file to temp directory
        draft_path = temp_dir / "draft_image.png"
        with open(draft_path, "wb") as f:
            shutil.copyfileobj(fig_draft_file.file, f)
        
        log.info(f"[figure_to_ppt_endpoint] Saved draft image to: {draft_path}")
        
        # Run pipeline with auto layout generation
        ppt_path = await run_figure_to_ppt_pipeline_with_auto_layout(
            fig_draft_path=str(draft_path),
            output_dir=str(temp_dir),
            mineru_port=mineru_port,
            mask_detail_level=mask_detail_level,
            figure_complex=figure_complex,
        )
        
        if not ppt_path or not Path(ppt_path).exists():
            raise HTTPException(
                status_code=500,
                detail="PPT generation failed: output file not found"
            )
        
        log.info(f"[figure_to_ppt_endpoint] Pipeline completed successfully")
        log.info(f"  PPT path: {ppt_path}")
        
        # Return PPT file
        return FileResponse(
            path=ppt_path,
            filename="output.pptx",
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[figure_to_ppt_endpoint] Pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}"
        )
    finally:
        # Cleanup: remove temp directory after a delay (in case file is still being served)
        # For now, we'll leave it for debugging, but in production you might want to schedule cleanup
        pass


@app.get("/api/figure-to-ppt/download")
async def download_ppt(ppt_path: str):
    """
    Download generated PPT file.
    
    Args:
        ppt_path: Path to the PPT file (relative to project root or absolute)
    """
    from pathlib import Path
    from fastapi.responses import FileResponse
    from fastapi import HTTPException
    from utils import get_project_root
    from utils.logger import get_logger
    
    log = get_logger(__name__)
    
    try:
        # Try as absolute path first
        ppt_file = Path(ppt_path)
        if not ppt_file.is_absolute():
            # Try relative to project root
            ppt_file = get_project_root() / ppt_path
        
        if not ppt_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"PPT file not found: {ppt_path}"
            )
        
        log.info(f"[download_ppt] Serving PPT file: {ppt_file}")
        
        return FileResponse(
            path=str(ppt_file),
            filename=ppt_file.name,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[download_ppt] Error serving PPT file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error serving PPT file: {str(e)}"
        )
