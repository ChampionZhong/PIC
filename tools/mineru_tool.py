"""
source code: https://github.com/OpenDCAI/DataFlow-Agent/blob/main/dataflow_agent/toolkits/imtool/mineru_tool.py
"""
# Before Using, do these:

# hf download opendatalab/MinerU2.5-2509-1.2B --local-dir opendatalab/MinerU2.5-2509-1.2B

# With vllm>=0.10.1, you can use following command to serve the model. The logits processor is used to support no_repeat_ngram_size sampling param, which can help the model to avoid generating repeated content.

# vllm serve opendatalab/MinerU2.5-2509-1.2B --host 127.0.0.1 --port <port> \
#   --logits-processors mineru_vl_utils:MinerULogitsProcessor
# If you are using vllm<0.10.1, no_repeat_ngram_size sampling param is not supported. You still can serve the model without logits processor:

# vllm serve models/MinerU2.5-2509-1.2B \
#     --host 127.0.0.1 \
#     --port 8001 \
#     --logits-processors mineru_vl_utils:MinerULogitsProcessor \
#     --gpu-memory-utilization 0.6

# vllm serve opendatalab/MinerU2.5-2509-1.2B --host 127.0.0.1 --port <port>


from pathlib import Path
from typing import Any, Dict, List, Sequence, Union, Optional
import os
import shutil
import subprocess
import base64
import asyncio
import httpx
import time

from PIL import Image

# Try to import MinerUClient for backward compatibility
try:
    from mineru_vl_utils import MinerUClient
    MINERU_CLIENT_AVAILABLE = True
except ImportError:
    MINERU_CLIENT_AVAILABLE = False
    MinerUClient = None


# ---------------------------------------
# Helper functions for MinerU configuration
# ---------------------------------------

def _get_mineru_config():
    """
    Get MinerU configuration from environment variables.
    Supports three modes:
    1. MinerU API mode (MINERU_API_KEY + MINERU_API_URL)
    2. MinerUClient with http-client backend
    3. MinerUClient with vllm-engine backend
    
    Returns:
        dict with config keys: api_key, api_url, use_api, backend, server_url, model_name, handle_equation_block
    """
    # Check for MinerU API mode (external API service)
    api_key = os.getenv("MINERU_API_KEY", "")
    api_url = os.getenv("MINERU_API_URL", "")
    use_api = bool(api_key and api_url)
    
    # Check for MinerUClient mode (local or remote vllm/http service)
    backend = os.getenv("MINERU_BACKEND", "http-client")  # Default to http-client
    server_url = os.getenv("MINERU_SERVER_URL", "")
    model_name = os.getenv("MINERU_MODEL_NAME", "mineru_vl_2509")
    handle_equation_block_str = os.getenv("MINERU_HANDLE_EQUATION_BLOCK", "false")
    handle_equation_block = handle_equation_block_str.lower() in ("true", "1", "yes")
    
    # Determine which mode to use
    # Priority: API mode > MinerUClient mode
    use_mineru_client = bool(server_url) and not use_api
    
    return {
        "api_key": api_key,
        "api_url": api_url,
        "use_api": use_api,
        "use_mineru_client": use_mineru_client,
        "backend": backend,
        "server_url": server_url,
        "model_name": model_name,
        "handle_equation_block": handle_equation_block,
    }


def _get_mineru_api_config():
    """Get MinerU API configuration (backward compatibility)"""
    config = _get_mineru_config()
    return config["api_key"], config["api_url"], config["use_api"]


def _encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class _NoProxyContext:
    """
    Context manager to temporarily disable proxy environment variables.
    
    This ensures MinerUClient (which uses httpx internally) does not use proxy settings.
    httpx clients read proxy settings from environment variables during initialization,
    so we must disable proxy BEFORE creating the client and keep it disabled throughout
    the client's lifetime.
    """
    def __init__(self):
        self.proxy_vars = [
            "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
            "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"
        ]
        self.saved_values = {}
    
    def __enter__(self):
        # Save current proxy environment variables
        for var in self.proxy_vars:
            self.saved_values[var] = os.environ.get(var)
            # Remove proxy environment variables to disable proxy
            if var in os.environ:
                del os.environ[var]
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original proxy environment variables
        for var, value in self.saved_values.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]


async def _call_mineru_api_async(image_path: str) -> List[Dict[str, Any]]:
    """
    Call MinerU API to extract content from image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of blocks with type, bbox, text, etc.
    """
    config = _get_mineru_config()
    
    if not config["use_api"]:
        raise ValueError(
            "MINERU_API_KEY and MINERU_API_URL must be set in environment variables. "
            "Please configure them in .env file."
        )
    
    api_key = config["api_key"]
    api_url = config["api_url"]
    
    # Encode image to base64
    image_b64 = _encode_image_to_base64(image_path)
    
    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # According to MinerU API docs (https://mineru.net/apiManage/docs)
    # Try multiple request formats to support different API versions
    
    # Disable proxy for MinerU API calls (MinerU should not go through proxy)
    async with httpx.AsyncClient(
        timeout=300.0,
        proxies=None,  # Explicitly disable proxy
        trust_env=False,  # Don't read proxy from environment variables
    ) as client:
        try:
            # Method 1: Try multipart/form-data with file upload
            try:
                with open(image_path, "rb") as f:
                    files = {"file": (Path(image_path).name, f, "image/png")}
                    data = {
                        "is_ocr": "true",
                        "enable_formula": "true",
                        "enable_table": "true",
                    }
                    # Remove Content-Type header for multipart
                    headers_multipart = {"Authorization": f"Bearer {api_key}"}
                    response = await client.post(
                        api_url,
                        headers=headers_multipart,
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Check response format
                    if result.get("code") == 0:
                        return _extract_blocks_from_response(result)
            except Exception as e1:
                # Method 2: Try JSON with base64 image
                try:
                    payload = {
                        "image": image_b64,
                        "is_ocr": True,
                        "enable_formula": True,
                        "enable_table": True,
                    }
                    response = await client.post(
                        api_url,
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if result.get("code") == 0:
                        # Check if it's a task-based API
                        if "task_id" in result.get("data", {}):
                            task_id = result["data"]["task_id"]
                            return await _poll_mineru_task(api_url, task_id, headers)
                        else:
                            return _extract_blocks_from_response(result)
                except Exception as e2:
                    # Method 3: Try direct URL format (if API supports file URLs)
                    raise RuntimeError(
                        f"MinerU API call failed. Tried multipart ({str(e1)}) and JSON ({str(e2)})"
                    )
                    
        except httpx.HTTPStatusError as e:
            error_text = e.response.text if hasattr(e.response, 'text') else str(e)
            raise RuntimeError(f"MinerU API HTTP error: {e.response.status_code} - {error_text}")
        except Exception as e:
            raise RuntimeError(f"MinerU API call failed: {str(e)}")


def _extract_blocks_from_response(result: dict) -> List[Dict[str, Any]]:
    """Extract blocks from MinerU API response"""
    # Try different response formats
    if "blocks" in result:
        blocks = result["blocks"]
        return blocks if isinstance(blocks, list) else []
    
    if "content" in result:
        content = result["content"]
        return content if isinstance(content, list) else []
    
    data = result.get("data", {})
    if "blocks" in data:
        blocks = data["blocks"]
        return blocks if isinstance(blocks, list) else []
    
    if "content" in data:
        content = data["content"]
        return content if isinstance(content, list) else []
    
    # If result is a list directly, return it
    if isinstance(result, list):
        return result
    
    # Fallback: return empty list
    return []


async def _poll_mineru_task(
    api_url: str,
    task_id: str,
    headers: dict,
    max_wait_time: int = 300,
    poll_interval: int = 2
) -> List[Dict[str, Any]]:
    """Poll MinerU task until completion"""
    task_result_url = f"{api_url.rstrip('/')}/{task_id}"
    start_time = time.time()
    
    # Disable proxy for MinerU API calls
    async with httpx.AsyncClient(
        timeout=300.0,
        proxies=None,  # Explicitly disable proxy
        trust_env=False,  # Don't read proxy from environment variables
    ) as client:
        while time.time() - start_time < max_wait_time:
            await asyncio.sleep(poll_interval)
            
            response = await client.get(task_result_url, headers=headers)
            response.raise_for_status()
            task_result = response.json()
            
            if task_result.get("code") == 0:
                task_data = task_result.get("data", {})
                task_state = task_data.get("state")
                
                if task_state == "done":
                    return _extract_blocks_from_response(task_data)
                elif task_state == "failed":
                    err_msg = task_data.get("err_msg", "Unknown error")
                    raise RuntimeError(f"MinerU task failed: {err_msg}")
                # Continue polling if state is "processing" or "pending"
            else:
                raise RuntimeError(f"MinerU API error: {task_result.get('msg', 'Unknown error')}")
        
        raise TimeoutError("MinerU task timeout: task did not complete within 5 minutes")


# ---------------------------------------
# 1. two_step_extract (sync)
# ---------------------------------------
def run_two_step_extract(image_path: str, port: int = None):
    """
    Synchronous call to MinerU two_step_extract.
    Supports three modes:
    1. MinerU API mode (MINERU_API_KEY + MINERU_API_URL)
    2. MinerUClient with http-client backend
    3. MinerUClient with vllm-engine backend
    """
    config = _get_mineru_config()
    
    if config["use_api"]:
        # Use API mode
        return asyncio.run(_call_mineru_api_async(image_path))
    elif config["use_mineru_client"]:
        # Use MinerUClient mode
        if not MINERU_CLIENT_AVAILABLE:
            raise ValueError(
                "mineru_vl_utils is required for MinerUClient mode. "
                "Please install it: pip install mineru-vl-utils"
            )
        image = Image.open(image_path)
        # Disable proxy BEFORE creating MinerUClient (httpx reads env vars during init)
        with _NoProxyContext():
            client = MinerUClient(
                model_name=config["model_name"],
                backend=config["backend"],
                server_url=config["server_url"],
                handle_equation_block=config["handle_equation_block"],
            )
            return client.two_step_extract(image)
    else:
        # Fallback to legacy local service mode (for backward compatibility)
        if not MINERU_CLIENT_AVAILABLE:
            raise ValueError(
                "Either configure MINERU_API_KEY/URL, MINERU_SERVER_URL, "
                "or install mineru_vl_utils and provide port for local service mode."
            )
        if port is None:
            raise ValueError("port is required for legacy local service mode")
        image = Image.open(image_path)
        # Disable proxy BEFORE creating MinerUClient (httpx reads env vars during init)
        with _NoProxyContext():
            client = MinerUClient(
                backend="http-client",
                server_url=f"http://127.0.0.1:{port}"
            )
            return client.two_step_extract(image)


# ---------------------------------------
# 2. batch_two_step_extract (sync)
# ---------------------------------------
def run_batch_two_step_extract(image_paths: list[str], port: int):
    """同步批量调用 MinerU two_step_extract，处理多张图片并返回结果列表。"""
    images = [Image.open(p) for p in image_paths]
    # Disable proxy BEFORE creating MinerUClient (httpx reads env vars during init)
    with _NoProxyContext():
        client = MinerUClient(
            backend="http-client",
            server_url=f"http://127.0.0.1:{port}"
        )
        return client.batch_two_step_extract(images)


# ---------------------------------------
# 3. aio_two_step_extract (async)
# ---------------------------------------
async def run_aio_two_step_extract(image_path: str, port: int = None, max_retries: int = 3, retry_delay: float = 2.0):
    """
    Async call to MinerU two_step_extract with retry mechanism.
    Supports three modes:
    1. MinerU API mode (MINERU_API_KEY + MINERU_API_URL)
    2. MinerUClient with http-client backend
    3. MinerUClient with vllm-engine backend
    
    Args:
        image_path: Path to the image to process
        port: Optional port for legacy local service mode
        max_retries: Maximum number of retries for transient errors (default: 3)
        retry_delay: Delay between retries in seconds (default: 2.0)
    """
    import asyncio
    
    config = _get_mineru_config()
    
    if config["use_api"]:
        # Use API mode with retry
        for attempt in range(max_retries):
            try:
                return await _call_mineru_api_async(image_path)
            except Exception as e:
                if attempt < max_retries - 1 and _is_retryable_error(e):
                    print(f"[run_aio_two_step_extract] Retryable error (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                raise
    elif config["use_mineru_client"]:
        # Use MinerUClient mode with retry
        if not MINERU_CLIENT_AVAILABLE:
            raise ValueError(
                "mineru_vl_utils is required for MinerUClient mode. "
                "Please install it: pip install mineru-vl-utils"
            )
        image = Image.open(image_path)
        
        # Disable proxy BEFORE creating MinerUClient (httpx reads env vars during init)
        with _NoProxyContext():
            client = MinerUClient(
                model_name=config["model_name"],
                backend=config["backend"],
                server_url=config["server_url"],
                handle_equation_block=config["handle_equation_block"],
            )
            
            # Retry logic for MinerUClient (proxy remains disabled throughout)
            for attempt in range(max_retries):
                try:
                    return await client.aio_two_step_extract(image)
                except Exception as e:
                    if attempt < max_retries - 1 and _is_retryable_error(e):
                        print(f"[run_aio_two_step_extract] Retryable error (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    raise
    else:
        # Fallback to legacy local service mode (for backward compatibility)
        if not MINERU_CLIENT_AVAILABLE:
            raise ValueError(
                "Either configure MINERU_API_KEY/URL, MINERU_SERVER_URL, "
                "or install mineru_vl_utils and provide port for local service mode."
            )
        if port is None:
            raise ValueError("port is required for legacy local service mode")
        image = Image.open(image_path)
        
        # Disable proxy BEFORE creating MinerUClient (httpx reads env vars during init)
        with _NoProxyContext():
            client = MinerUClient(
                backend="http-client",
                server_url=f"http://127.0.0.1:{port}"
            )
            
            # Retry logic for legacy local service mode (proxy remains disabled throughout)
            for attempt in range(max_retries):
                try:
                    return await client.aio_two_step_extract(image)
                except Exception as e:
                    if attempt < max_retries - 1 and _is_retryable_error(e):
                        print(f"[run_aio_two_step_extract] Retryable error (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    raise


def _is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable (e.g., 502 Bad Gateway, 503 Service Unavailable).
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    error_str = str(error).lower()
    retryable_status_codes = ["502", "503", "504", "429"]  # Bad Gateway, Service Unavailable, Gateway Timeout, Too Many Requests
    
    # Check for retryable status codes in error message
    for code in retryable_status_codes:
        if code in error_str:
            return True
    
    # Check for connection errors
    if "connection" in error_str or "timeout" in error_str:
        return True
    
    # Check if it's a ServerError from mineru_vl_utils
    try:
        from mineru_vl_utils.vlm_client.base_client import ServerError
        if isinstance(error, ServerError):
            # Check status code if available
            if hasattr(error, "status_code"):
                return error.status_code in [502, 503, 504, 429]
            # Otherwise check error message
            return any(code in str(error) for code in retryable_status_codes)
    except ImportError:
        pass
    
    return False


# ---------------------------------------
# 4. aio_batch_two_step_extract (async)
# ---------------------------------------
async def run_aio_batch_two_step_extract(image_paths: list[str], port: int = None):
    """
    Async batch call to MinerU two_step_extract.
    Supports three modes:
    1. MinerU API mode (MINERU_API_KEY + MINERU_API_URL)
    2. MinerUClient with http-client backend
    3. MinerUClient with vllm-engine backend
    """
    config = _get_mineru_config()
    images = [Image.open(p) for p in image_paths]
    
    if config["use_mineru_client"]:
        if not MINERU_CLIENT_AVAILABLE:
            raise ValueError(
                "mineru_vl_utils is required for MinerUClient mode. "
                "Please install it: pip install mineru-vl-utils"
            )
        # Disable proxy BEFORE creating MinerUClient (httpx reads env vars during init)
        with _NoProxyContext():
            client = MinerUClient(
                model_name=config["model_name"],
                backend=config["backend"],
                server_url=config["server_url"],
                handle_equation_block=config["handle_equation_block"],
            )
            return await client.aio_batch_two_step_extract(images)
    else:
        # Fallback to legacy local service mode
        if not MINERU_CLIENT_AVAILABLE:
            raise ValueError("mineru_vl_utils is required")
        if port is None:
            raise ValueError("port is required for legacy local service mode")
        # Disable proxy BEFORE creating MinerUClient (httpx reads env vars during init)
        with _NoProxyContext():
            client = MinerUClient(
                backend="http-client",
                server_url=f"http://127.0.0.1:{port}"
            )
            return await client.aio_batch_two_step_extract(images)


# ---------------------------------------
# 5. 根据 MinerU bbox & type 裁剪原图
# ---------------------------------------
def crop_mineru_blocks_by_type(
    image_path: str,
    blocks: List[Dict[str, Any]],
    target_type: Optional[Union[str, Sequence[str]]] = None,
    output_dir: str = "",
    prefix: str = "",
) -> List[str]:
    """
    根据 MinerU two_step_extract / aio_two_step_extract 的结构化结果，
    按指定 type 的 bbox 从整张图片中裁剪子图并保存到输出目录。

    参数:
        image_path: 原始图片路径 (如技术路线图 PNG)
        blocks: MinerU 返回的 list[dict] 结果
        target_type: 需要裁剪的块类型，如 "title" / "text" / "image" / "footer"
        output_dir: 输出目录路径，不存在会自动创建
        prefix: 输出文件名前缀，可选

    返回:
        所有成功保存的裁剪图片的绝对路径列表
    """
    img = Image.open(image_path)
    width, height = img.size

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []

    # target_type 为 None 时不过滤，返回所有 block
    if target_type is None:
        target_types = None
    elif isinstance(target_type, str):
        target_types = {target_type}
    else:
        target_types = set(target_type)

    for idx, block in enumerate(blocks):
        block_type = block.get("type")
        # 只有在显式指定了 target_type 时才进行过滤
        if target_types is not None and block_type not in target_types:
            continue

        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1_norm, y1_norm, x2_norm, y2_norm = bbox

        # 将归一化坐标 [0,1] 转为像素坐标，并做边界裁剪
        left = max(0, min(width, int(round(x1_norm * width))))
        top = max(0, min(height, int(round(y1_norm * height))))
        right = max(0, min(width, int(round(x2_norm * width))))
        bottom = max(0, min(height, int(round(y2_norm * height))))

        # 无效 bbox 跳过
        if right <= left or bottom <= top:
            continue

        cropped = img.crop((left, top, right, bottom))

        # 使用实际 block_type 命名，便于区分不同类型
        safe_block_type = block_type or "unknown"
        filename = f"{prefix}{safe_block_type}_{idx}.png"
        out_path = out_dir / filename
        cropped.save(out_path)

        saved_paths.append(str(out_path.resolve()))

    return saved_paths

def run_mineru_pdf_extract(
    pdf_path: str,
    output_dir: str = "",
    source: str = "modelscope",
    mineru_executable: Optional[str] = None,
):
    """
    使用 MinerU 命令行方式提取 PDF 中的结构化内容，

    参数:
        pdf_path: PDF 文件路径
        output_dir: 输出目录路径，不存在会自动创建
        source: 下载模型的源，可选 modelscope、huggingface
        mineru_executable: mineru 可执行文件路径，
            - 不传时：优先从环境变量 MINERU_CMD 中读取，
              若没有则从 PATH 中查找 'mineru'
            - 传入绝对路径时：直接使用该路径

    返回:
        解析的所有图片、markdown格式的内容
    """
    # 1. 解析 mineru 可执行路径
    if mineru_executable is None:
        mineru_executable = (
            os.environ.get("MINERU_CMD")  # 环境变量优先
            or shutil.which("mineru")     # 当前 env 的命令
        )
        if mineru_executable is None:
            raise RuntimeError(
                "未找到 `mineru` 可执行文件，请确保：\n"
                "1) 已在当前环境安装 MinerU，并且 `mineru` 在 PATH 中；或\n"
                "2) 设置环境变量 MINERU_CMD 指向 mineru 可执行文件；或\n"
                "3) 调用 run_mineru_pdf_extract 时显式传入 mineru_executable 参数。"
            )

    mineru_cmd = [
        str(mineru_executable),
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "--source",
        source,
    ]

    # 2. 可选：自动创建 output_dir
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 3. 执行命令
    subprocess.run(
        mineru_cmd,
        shell=False,
        check=True,
        text=True,
        stderr=None,
        stdout=None,
    )



def crop_mineru_blocks_with_meta(
    image_path: str,
    blocks: List[Dict[str, Any]],
    target_type: Optional[Union[str, Sequence[str]]] = None,
    output_dir: str = "",
    prefix: str = "",
) -> List[Dict[str, Any]]:
    """
    与 ``crop_mineru_blocks_by_type`` 类似，但返回包含元信息的列表，
    方便后续根据 MinerU 的 bbox 在 PPT 中按比例还原布局。

    返回的每个元素包含:
        - block_index: 在原始 blocks 列表中的索引
        - type: MinerU 块类型
        - bbox: 原始归一化 bbox [x1, y1, x2, y2]
        - png_path: 裁剪得到的小图 PNG 绝对路径
    """
    img = Image.open(image_path)
    width, height = img.size

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    # target_type 为 None 时不过滤，返回所有 block
    if target_type is None:
        target_types = None
    elif isinstance(target_type, str):
        target_types = {target_type}
    else:
        target_types = set(target_type)

    for idx, block in enumerate(blocks):
        block_type = block.get("type")
        # 只有在显式指定了 target_type 时才进行过滤
        if target_types is not None and block_type not in target_types:
            continue

        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1_norm, y1_norm, x2_norm, y2_norm = bbox

        # 将归一化坐标 [0,1] 转为像素坐标，并做边界裁剪
        left = max(0, min(width, int(round(x1_norm * width))))
        top = max(0, min(height, int(round(y1_norm * height))))
        right = max(0, min(width, int(round(x2_norm * width))))
        bottom = max(0, min(height, int(round(y2_norm * height))))

        if right <= left or bottom <= top:
            continue

        cropped = img.crop((left, top, right, bottom))

        safe_block_type = block_type or "unknown"
        filename = f"{prefix}{safe_block_type}_{idx}.png"
        out_path = out_dir / filename
        cropped.save(out_path)

        results.append(
            {
                "block_index": idx,
                "type": block_type,
                "bbox": bbox,
                "png_path": str(out_path.resolve()),
            }
        )

    return results


def svg_to_emf(svg_path: str, emf_path: str, dpi: int = 600) -> str:
    """
    使用 Inkscape 将 SVG 文件转换为 EMF 矢量图，返回生成的 EMF 路径。
    使用 Inkscape 将 SVG 转换为 EMF 矢量图。

    依赖
    ----
    - 优先使用项目目录下的 Inkscape appimage (./models/Inkscape/)
    - 如果不存在，则使用系统 PATH 中的 `inkscape` 命令

    参数
    ----
    svg_path:
        输入 SVG 文件路径。
    emf_path:
        输出 EMF 文件路径。

    返回
    ----
    str
        生成的 EMF 文件的绝对路径。

    异常
    ----
    FileNotFoundError
        当输入 SVG 文件不存在时。
    RuntimeError
        当 Inkscape 调用失败或未生成输出文件时。
    """
    from utils import get_project_root
    
    svg_p = Path(svg_path)
    if not svg_p.exists():
        raise FileNotFoundError(f"输入 SVG 不存在: {svg_p}")

    emf_p = Path(emf_path)
    emf_p.parent.mkdir(parents=True, exist_ok=True)

    # Try to find Inkscape appimage in project directory first
    inkscape_cmd = None
    project_root = get_project_root()
    inkscape_appimage = project_root / "models" / "Inkscape" / "Inkscape-ebf0e94-x86_64.AppImage"
    
    if inkscape_appimage.exists():
        # Make sure appimage is executable
        os.chmod(inkscape_appimage, 0o755)
        inkscape_cmd = str(inkscape_appimage)
    else:
        # Fallback to system inkscape
        inkscape_cmd = "inkscape"

    try:
        # inkscape input.svg --export-filename=output.emf
        result = subprocess.run(
            [
                inkscape_cmd,
                str(svg_p),
                "--export-filename",
                str(emf_p),
                "--export-text-to-path",
                f"--export-dpi={dpi}"
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"调用 Inkscape 失败：未找到 Inkscape 可执行文件。\n"
            f"已尝试路径: {inkscape_appimage}\n"
            f"请确保 Inkscape appimage 存在于 ./models/Inkscape/ 目录，"
            f"或在系统 PATH 中安装 Inkscape。"
        ) from e

    if result.returncode != 0:
        raise RuntimeError(
            f"Inkscape 转换失败，返回码 {result.returncode}：\n"
            f"使用的命令: {inkscape_cmd}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    if not emf_p.exists():
        raise RuntimeError(f"Inkscape 运行后未发现输出 EMF 文件: {emf_p}")

    return str(emf_p.resolve())


# ---------------------------------------
# 6. 递归 MinerU 拆图 + 坐标映射 (HTTP 版)
# ---------------------------------------
def _crop_image_by_norm_bbox(
    image_path: str,
    bbox: Sequence[float],
    output_dir: Union[str, Path],
    prefix: str = "",
    index: int = 0,
) -> str:
    """
    按归一化 bbox [x1,y1,x2,y2] 从 image_path 裁剪出子图并保存，返回绝对路径。
    """
    img = Image.open(image_path)
    width, height = img.size

    x1_norm, y1_norm, x2_norm, y2_norm = bbox
    left = max(0, min(width, int(round(x1_norm * width))))
    top = max(0, min(height, int(round(y1_norm * height))))
    right = max(0, min(width, int(round(x2_norm * width))))
    bottom = max(0, min(height, int(round(y2_norm * height))))

    if right <= left or bottom <= top:
        raise ValueError(f"Invalid bbox after clamp: {bbox}")

    cropped = img.crop((left, top, right, bottom))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    filename = f"{prefix}{stem}_{index}.png"
    out_path = out_dir / filename
    cropped.save(out_path)

    return str(out_path.resolve())


async def recursive_mineru_layout(
    image_path: str,
    port: int = None,
    max_depth: int = 2,
    current_depth: int = 0,
    output_dir: Optional[Union[str, Path]] = None,
    block_types_for_subimage: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    使用 MinerU HTTP two_step_extract，递归拆图并将所有最底层块映射到
    最顶层图的归一化坐标系。

    返回的每个元素形如:
        {
            "type": str,
            "bbox": [x1, y1, x2, y2],   # 相对于最顶层图的归一化坐标
            "png_path": str | None,     # 对应子图路径（图像/表格等）
            "text": str | None,         # 文本内容（若有）
            "depth": int,               # 所在递归深度
        }
    """
    if current_depth > max_depth:
        return []

    # 默认在原图同目录下创建一个 mineru_recursive 子目录
    if output_dir is None:
        base = Path(image_path).with_suffix("")
        output_dir = base.parent / f"{base.stem}_mineru_recursive"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 默认哪些类型会继续拆成子图
    if block_types_for_subimage is None:
        block_types_for_subimage = ["image", "img", "table", "figure"]

    # 1. 当前层 MinerU 调用
    blocks = await run_aio_two_step_extract(image_path=image_path, port=port)

    leaf_items: List[Dict[str, Any]] = []

    # blocks 结构假定为 List[Dict]，包含 type / bbox / text 等
    for idx, blk in enumerate(blocks):
        blk_type = blk.get("type")
        bbox = blk.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        # 保证归一化 bbox 在 [0,1] 内，大致裁剪
        x1, y1, x2, y2 = bbox
        x1 = max(0.0, min(1.0, float(x1)))
        y1 = max(0.0, min(1.0, float(y1)))
        x2 = max(0.0, min(1.0, float(x2)))
        y2 = max(0.0, min(1.0, float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        norm_bbox = [x1, y1, x2, y2]

        # 如果是需要继续拆的图块类型，裁剪子图并递归
        if blk_type in block_types_for_subimage and current_depth < max_depth:
            try:
                sub_img_path = _crop_image_by_norm_bbox(
                    image_path=image_path,
                    bbox=norm_bbox,
                    output_dir=out_dir / "sub_images",
                    prefix=f"depth{current_depth}_blk{idx}_",
                    index=idx,
                )
            except Exception:
                # 裁剪失败则当成叶子块处理
                leaf_items.append(
                    {
                        "type": blk_type,
                        "bbox": norm_bbox,
                        "png_path": None,
                        "text": blk.get("text") or blk.get("content"),
                        "depth": current_depth,
                    }
                )
                continue

            # 子图内部是完整的 [0,1] 坐标系，需要映射回当前图的 norm_bbox
            sub_items = await recursive_mineru_layout(
                image_path=sub_img_path,
                port=port,  # port can be None if using API mode
                max_depth=max_depth,
                current_depth=current_depth + 1,
                output_dir=out_dir,
                block_types_for_subimage=block_types_for_subimage,
            )

            pw = norm_bbox[2] - norm_bbox[0]
            ph = norm_bbox[3] - norm_bbox[1]
            for si in sub_items:
                sb = si.get("bbox")
                if not sb or len(sb) != 4:
                    continue
                sx1, sy1, sx2, sy2 = sb
                nx1 = norm_bbox[0] + sx1 * pw
                ny1 = norm_bbox[1] + sy1 * ph
                nx2 = norm_bbox[0] + sx2 * pw
                ny2 = norm_bbox[1] + sy2 * ph
                si["bbox"] = [nx1, ny1, nx2, ny2]
                leaf_items.append(si)
        else:
            # 文本或其他不再下钻的类型，直接当作叶子
            leaf_items.append(
                {
                    "type": blk_type,
                    "bbox": norm_bbox,
                    "png_path": None,
                    "text": blk.get("text") or blk.get("content"),
                    "depth": current_depth,
                }
            )

    return leaf_items
