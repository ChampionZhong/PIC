"""
Reference: https://github.com/OpenDCAI/DataFlow-Agent
"""

"""
Image editing tool - for generating empty template images
Support Gemini model image editing functionality
"""

import os
import base64
import re
import json
import asyncio
from typing import Tuple, Optional
import httpx
import requests
from enum import Enum
from pathlib import Path

# ============ Configuration and utility functions ============

class Provider(str, Enum):
    APIYI = "apiyi"
    LOCAL_123 = "local_123"
    OTHER = "other"

_B64_RE = re.compile(r"[A-Za-z0-9+/=]+")


def detect_provider(api_url: str) -> Provider:
    """Detect provider based on api_url"""
    if "api.apiyi.com" in api_url:
        return Provider.APIYI
    if "123.129.219.111" in api_url:
        return Provider.LOCAL_123
    return Provider.OTHER


def _encode_image_to_base64(image_path: str) -> Tuple[str, str]:
    """Read image and encode to Base64, return (base64_string, format)"""
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    
    ext = image_path.rsplit(".", 1)[-1].lower()
    if ext in {"jpg", "jpeg"}:
        fmt = "jpeg"
    elif ext == "png":
        fmt = "png"
    else:
        raise ValueError(f"Unsupported image format: {ext}")
    
    return b64, fmt


def extract_base64(s: str) -> str:
    """Extract longest continuous Base64 string from string"""
    s = "".join(s.split())
    matches = _B64_RE.findall(s)
    return max(matches, key=len) if matches else ""


# ============ Model detection functions ============

def _is_gemini_model(model: str) -> bool:
    """Check if model is Gemini series"""
    return 'gemini' in model.lower()


def is_gemini_25(model: str) -> bool:
    """Check if model is Gemini 2.5 series"""
    return "gemini-2.5" in model.lower()


def is_gemini_3_pro(model: str) -> bool:
    """Check if model is Gemini 3 Pro series"""
    return "gemini-3-pro" in model.lower()


# ============ HTTP request functions ============

async def _post_raw(
    url: str,
    api_key: str,
    payload: dict,
    timeout: int,
) -> dict:
    """
    Uniform POST request using requests library (same as run_painter)
    This ensures consistency with other Gemini API calls and better proxy support
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    print(f"POST {url}")
    
    # Use requests.request in thread pool (same as run_painter)
    # This ensures consistent behavior with proxy settings
    try:
        response = await asyncio.to_thread(
            requests.request,
            "POST",
            url,
            data=json.dumps(payload),
            headers=headers,
            timeout=timeout
        )
        
        print(f"status={response.status_code}")
        response.raise_for_status()
        
        # Check if response is HTML (common error case)
        response_text = response.text.strip()
        if response_text.startswith("<!doctype html>") or response_text.startswith("<html"):
            error_msg = (
                f"Received HTML response instead of JSON. This usually means the API endpoint is incorrect.\n"
                f"URL: {url}\n"
                f"Response preview: {response_text[:500]}..."
            )
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        
        # Check if response is empty
        if not response_text:
            error_msg = (
                f"Received empty response from API.\n"
                f"URL: {url}\n"
                f"Status code: {response.status_code}"
            )
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        
        # Try to parse JSON
        try:
            return response.json()
        except json.JSONDecodeError as json_err:
            error_msg = (
                f"Failed to parse JSON response.\n"
                f"Error: {json_err}\n"
                f"Response text (first 500 chars): {response_text[:500]}...\n"
                f"Full response length: {len(response_text)}"
            )
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from json_err
            
    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if hasattr(e, 'response') and hasattr(e.response, 'text') else str(e)
        error_msg = f"HTTP {e.response.status_code} error: {error_text}"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg) from e
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}. URL: {url}"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg) from e


async def _post_chat_completions(
    api_url: str,
    api_key: str,
    payload: dict,
    timeout: int,
) -> dict:
    """OpenAI 兼容的 /chat/completions POST"""
    url = f"{api_url.rstrip('/')}/chat/completions"
    return await _post_raw(url, api_key, payload, timeout)


# ============ Gemini 编辑请求构建 ============

def build_gemini_edit_request(
    api_url: str,
    model: str,
    prompt: str,
    aspect_ratio: str,
    b64: str,
    fmt: str,
    resolution: str = "2K",
) -> tuple[str, dict]:
    """
    根据服务商 + 模型构造图像编辑请求的 (url, payload)
    """
    provider = detect_provider(api_url)
    base = api_url.rstrip("/")
    
    # 1) APIYI + Gemini 2.5
    if provider is Provider.APIYI and is_gemini_25(model) and aspect_ratio != "1:1":
        url = "https://api.apiyi.com/v1beta/models/gemini-2.5-flash-image:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": f"image/{fmt}",
                                "data": b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                },
            },
        }
        return url, payload
    
    # 2) APIYI + Gemini 3 Pro
    if provider is Provider.APIYI and is_gemini_3_pro(model):
        url = "https://api.apiyi.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": f"image/{fmt}",
                                "data": b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution,
                },
            },
        }
        return url, payload
    
    # 3) LOCAL_123 + Gemini 3 Pro
    if provider is Provider.LOCAL_123 and is_gemini_3_pro(model):
        url = f"{base}/chat/completions"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{fmt};base64,{b64}",
                        },
                    },
                ],
            }
        ]
        payload = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "image"},
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        return url, payload
    
    # 4) LOCAL_123 + Gemini 2.5
    if provider is Provider.LOCAL_123 and is_gemini_25(model):
        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": f"image/{fmt}",
                                "data": b64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "width": 1920,
                "height": 1080,
                "quality": "high",
            },
        }
        return url, payload
    
    # 5) 其他服务商 - 根据模型类型选择正确的端点
    # 对于 gemini-3-pro-image-preview，使用和 run_painter 相同的端点
    if is_gemini_3_pro(model):
        # Use the same URL building logic as run_painter
        from urllib.parse import urlparse
        
        default_endpoint = "/v1beta/models/gemini-3-pro-image-preview:generateContent/"
        
        parsed = urlparse(base)
        if parsed.path:
            base_path = parsed.path.rstrip('/')
            if "gemini-3-pro-image-preview:generateContent" in base_path:
                endpoint = base_path + ("/" if not base_path.endswith("/") else "")
            elif base_path.endswith("/v1beta"):
                endpoint = base_path + "/models/gemini-3-pro-image-preview:generateContent/"
            elif base_path == "" or base_path == "/":
                endpoint = default_endpoint
            else:
                endpoint = base_path.rstrip('/') + default_endpoint
        else:
            endpoint = default_endpoint
        
        url = base + endpoint
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": f"image/{fmt}",
                            "data": b64,
                        }
                    }
                ]
            }],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution,
                }
            }
        }
        return url, payload
    
    # 6) 其他情况 - OpenAI 兼容格式（fallback）
    url = f"{base}/chat/completions"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{fmt};base64,{b64}",
                    },
                },
            ],
        }
    ]
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "image"},
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    return url, payload


# ============ Gemini 图像编辑调用 ============

async def call_gemini_image_edit_async(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: str,
    timeout: int = 120,
    aspect_ratio: str = "1:1",
    resolution: str = "2K",
) -> dict:
    """
    调用 Gemini 图像编辑 API
    返回响应 JSON
    """
    provider = detect_provider(api_url)
    
    # LOCAL_123 + Gemini 2.5 特殊处理
    if provider is Provider.LOCAL_123 and is_gemini_25(model):
        b64, fmt = _encode_image_to_base64(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{fmt};base64,{b64}"},
                    },
                ],
            }
        ]
        payload = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "image"},
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        return await _post_chat_completions(api_url, api_key, payload, timeout)
    
    # 其他情况
    b64, fmt = _encode_image_to_base64(image_path)
    url, payload = build_gemini_edit_request(
        api_url, model, prompt, aspect_ratio, b64, fmt, resolution
    )
    
    # Log the URL being used for debugging
    print(f"[call_gemini_image_edit_async] Using URL: {url}")
    print(f"[call_gemini_image_edit_async] Provider: {provider}")
    print(f"[call_gemini_image_edit_async] Model: {model}")
    
    try:
        return await _post_raw(url, api_key, payload, timeout)
    except RuntimeError as e:
        # If we get HTML response, it means the endpoint is wrong
        error_msg = str(e)
        if "Failed to parse JSON" in error_msg and "<!doctype html>" in error_msg.lower():
            raise RuntimeError(
                f"API endpoint returned HTML instead of JSON. "
                f"This usually means the API URL or endpoint path is incorrect. "
                f"URL used: {url}. "
                f"Please check your GEMINI_API_URL configuration. "
                f"Expected: API endpoint that returns JSON. "
                f"Got: HTML page (likely a management interface)."
            ) from e
        raise


# ============ 统一接口 ============

async def generate_layout_image_async(
    prompt: str,
    image_path: str,
    save_path: str,
    api_url: str,
    api_key: str,
    model: str,
    aspect_ratio: str = "16:9",
    resolution: str = "2K",
    timeout: int = 300,
) -> str:
    """
    生成空模板图的统一接口
    
    参数:
        prompt: 编辑提示词（如 TEMPLATE_EDIT_PROMPT）
        image_path: 输入图像路径（带内容的图）
        save_path: 保存路径
        api_url: API 地址
        api_key: API Key
        model: 模型名称（如 "gemini-3-pro-image-preview"）
        aspect_ratio: 宽高比（如 "16:9", "1:1", "9:16"）
        resolution: 分辨率（Gemini-3 Pro 专用，可选: "1K", "2K", "4K"）
        timeout: 超时时间（秒）
    
    返回:
        生成的图像 base64 字符串
    """
    if not _is_gemini_model(model):
        raise ValueError(f"当前仅支持 Gemini 模型，不支持: {model}")
    
    # 调用编辑 API
    raw_data = await call_gemini_image_edit_async(
        api_url=api_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        image_path=image_path,
        timeout=timeout,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )
    
    # 解析响应，提取 base64
    data = raw_data
    
    # Debug: 打印响应结构（截断以避免日志过大）
    print(f"[generate_layout_image_async] Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    print(f"[generate_layout_image_async] Response preview: {str(data)[:1000]}...")
    
    # 1) Gemini candidates 结构
    if "candidates" in data:
        try:
            candidates = data["candidates"]
            if not candidates:
                raise RuntimeError("candidates 为空")
            content = candidates[0]["content"]
            parts = content["parts"]
            inline_data = parts[0]["inlineData"]
            b64 = inline_data["data"]
        except Exception as e:
            print(f"[generate_layout_image_async] ERROR: 解析 Gemini candidates 结构失败: {e}")
            print(f"[generate_layout_image_async] 完整响应（截断）: {str(data)[:2000]}")
            raise RuntimeError(f"解析 Gemini candidates 结构失败: {e}") from e
    
    # 2) OpenAI 兼容 choices 结构
    elif "choices" in data:
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                b64 = extract_base64(content)
            elif isinstance(content, list):
                joined = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
                b64 = extract_base64(joined)
            else:
                raise RuntimeError(f"不支持的 content 类型: {type(content)}")
            
            if not b64:
                raise RuntimeError("从响应中未提取到 Base64")
        except Exception as e:
            print(f"[generate_layout_image_async] ERROR: 解析 OpenAI 兼容响应失败: {e}")
            print(f"[generate_layout_image_async] 完整响应（截断）: {str(data)[:2000]}")
            raise RuntimeError(f"解析 OpenAI 兼容响应失败: {e}") from e
    
    # 3) 尝试直接从响应中提取 base64（某些 API 可能返回不同的结构）
    else:
        print(f"[generate_layout_image_async] WARNING: 未知的响应结构，尝试从整个响应中提取 base64")
        print(f"[generate_layout_image_async] 响应内容（截断）: {str(data)[:2000]}")
        
        # 尝试从整个响应字符串中提取 base64
        response_str = json.dumps(data) if isinstance(data, dict) else str(data)
        b64 = extract_base64(response_str)
        
        if not b64:
            error_msg = (
                f"未知的响应结构：缺少 candidates / choices 字段，且无法提取 base64。\n"
                f"响应 keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}\n"
                f"响应预览: {str(data)[:2000]}"
            )
            print(f"[generate_layout_image_async] ERROR: {error_msg}")
            raise RuntimeError(error_msg)
    
    # 保存图像
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(b64))
    
    print(f"空模板图已保存至: {save_path}")
    return b64