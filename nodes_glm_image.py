import threading
from typing import Any, Dict, List

import importlib
import importlib.util
import inspect
import traceback
import numpy as np
import torch
from PIL import Image
import os

try:
    import comfy.model_management as mm
except Exception:
    mm = None

try:
    import comfy.utils as comfy_utils
except Exception:
    comfy_utils = None

try:
    import folder_paths
except Exception:
    folder_paths = None


_PIPELINE_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()

# Debug controls (isolated from normal runtime path)
# Set GLM_IMAGE_DEBUG=1 to enable debug logs.
# Set GLM_IMAGE_DEBUG_STRICT=1 to turn suspicious outputs into runtime errors.
_GLM_DEBUG_ENABLED = True
_GLM_DEBUG_STRICT = os.getenv("GLM_IMAGE_DEBUG_STRICT", "0") == "1"


def _glm_debug(message: str) -> None:
    if _GLM_DEBUG_ENABLED:
        print(f"[ComfyUI-GLM-Image][DEBUG] {message}")


def _glm_debug_check_images(images: List[Image.Image], context: str) -> None:
    if not _GLM_DEBUG_ENABLED or not images:
        return

    stats = []
    has_non_finite = False
    near_black_count = 0
    for i, img in enumerate(images):
        arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        finite_mask = np.isfinite(arr)
        finite_ratio = float(finite_mask.mean())
        if finite_ratio < 1.0:
            has_non_finite = True
        safe_arr = np.where(finite_mask, arr, 0.0)
        v_min = float(safe_arr.min())
        v_max = float(safe_arr.max())
        v_mean = float(safe_arr.mean())
        if v_max < 1e-4:
            near_black_count += 1
        stats.append(
            f"img[{i}]: min={v_min:.6f}, max={v_max:.6f}, mean={v_mean:.6f}, finite={finite_ratio:.6f}"
        )

    _glm_debug(f"{context} output stats: " + " | ".join(stats))

    suspicious = has_non_finite or (near_black_count == len(images))
    if suspicious:
        reason = []
        if has_non_finite:
            reason.append("non-finite values detected")
        if near_black_count == len(images):
            reason.append("all outputs are near-black")
        msg = f"{context} suspicious output: {', '.join(reason)}"
        _glm_debug(msg)
        if _GLM_DEBUG_STRICT:
            raise RuntimeError(
                f"GLM-Image 调试检查失败：{msg}。"
                "可关闭严格模式：GLM_IMAGE_DEBUG_STRICT=0"
            )


def _ensure_dependency() -> None:
    required_modules = [
        "transformers",
        "diffusers",
        "accelerate",
        "safetensors",
        "sentencepiece",
    ]

    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "GLM-Image 节点缺少依赖："
            + ", ".join(missing)
            + "\n请先安装：\n"
            "pip install git+https://github.com/huggingface/transformers.git\n"
            "pip install git+https://github.com/huggingface/diffusers.git\n"
            "pip install accelerate safetensors sentencepiece"
        )

    # 依赖存在但导入仍可能失败（例如版本冲突、损坏或环境变量问题），需要保留真实异常。
    try:
        importlib.import_module("diffusers")
        importlib.import_module("transformers")
    except Exception as e:
        raise RuntimeError(
            "GLM-Image 依赖已安装，但导入失败。\n"
            "这通常是版本冲突、环境不一致或包损坏导致。\n"
            f"原始异常：{type(e).__name__}: {e}"
        ) from e

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    if mm is not None:
        torch_device = str(mm.get_torch_device())
        if "cuda" in torch_device:
            return "cuda"
        if "mps" in torch_device:
            return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(dtype: str, device: str):
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16

    # auto
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # GLM-Image 在部分显卡/驱动组合下 fp16 可能出现数值不稳定，优先保证稳定性。
        return torch.float32
    return torch.float32


def _patch_processor_apply_chat_template(pipe: Any) -> None:
    processor = getattr(pipe, "processor", None)
    if processor is None:
        return

    processor_cls = processor.__class__
    if getattr(processor_cls, "_comfy_glm_apply_chat_template_patched", False):
        setattr(pipe, "_comfy_glm_apply_chat_template_patched", True)
        return

    original = getattr(processor_cls, "apply_chat_template", None)
    if original is None:
        return

    try:
        sig = inspect.signature(original)
    except Exception:
        return

    if "processor_kwargs" not in sig.parameters:
        return

    def _wrapped_apply_chat_template(self, *args, **kwargs):
        processor_arg_keys = {"target_h", "target_w", "padding"}
        migrated = {}
        for key in list(kwargs.keys()):
            if key in processor_arg_keys:
                migrated[key] = kwargs.pop(key)

        if migrated:
            processor_kwargs = dict(kwargs.pop("processor_kwargs", {}) or {})
            for key, value in migrated.items():
                processor_kwargs.setdefault(key, value)
            kwargs["processor_kwargs"] = processor_kwargs
            _glm_debug(
                "patched apply_chat_template: moved args into processor_kwargs "
                f"(target={processor_kwargs.get('target_h')}x{processor_kwargs.get('target_w')}, "
                f"padding={processor_kwargs.get('padding')})"
            )
        return original(self, *args, **kwargs)

    setattr(processor_cls, "apply_chat_template", _wrapped_apply_chat_template)
    setattr(processor_cls, "_comfy_glm_apply_chat_template_patched", True)
    setattr(pipe, "_comfy_glm_apply_chat_template_patched", True)


def _align_to_32(value: int) -> int:
    return max(32, int(round(value / 32.0) * 32))


def _validate_size_divisible_by_32(width: int, height: int) -> None:
    if width % 32 != 0 or height % 32 != 0:
        raise ValueError(
            f"GLM-Image 要求 width/height 必须是 32 的倍数，当前 width={width}, height={height}"
        )


def _tensor_to_pil_batch(image: torch.Tensor) -> List[Image.Image]:
    if image.dim() == 3:
        image = image.unsqueeze(0)

    if image.dim() != 4:
        raise ValueError(
            f"输入图像维度错误，期望 4 维(B,H,W,C)，实际: {tuple(image.shape)}，dtype={image.dtype}"
        )

    pil_images: List[Image.Image] = []
    image_cpu = image.detach().cpu().clamp(0.0, 1.0)
    for i in range(image_cpu.shape[0]):
        arr = (image_cpu[i].numpy() * 255.0).astype(np.uint8)
        if arr.shape[-1] == 4:
            pil_images.append(Image.fromarray(arr, mode="RGBA"))
        else:
            if arr.shape[-1] != 3:
                raise ValueError(
                    f"输入图像通道数仅支持 3/4，实际: {arr.shape[-1]}，shape={arr.shape}"
                )
            pil_images.append(Image.fromarray(arr, mode="RGB"))
    return pil_images


def _pil_to_tensor_batch(images: List[Image.Image]) -> torch.Tensor:
    if not images:
        raise ValueError("模型未返回任何图像（result.images 为空）")
    tensors = []
    for img in images:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    return torch.stack(tensors, dim=0)


def _decoded_batch_metrics(decoded: torch.Tensor):
    if decoded.dim() != 4:
        raise ValueError(f"VAE decode 输出维度错误，期望 4 维(B,C,H,W)，实际: {tuple(decoded.shape)}")

    safe_decoded = torch.nan_to_num(decoded, nan=0.0, posinf=0.0, neginf=0.0)
    flat = safe_decoded.flatten(1)
    finite_ratio = torch.isfinite(decoded).flatten(1).float().mean(dim=1)
    std = flat.std(dim=1, unbiased=False)
    value_range = flat.max(dim=1).values - flat.min(dim=1).values
    near_black_ratio = (flat <= (1.0 / 255.0)).float().mean(dim=1)

    bad_non_finite = finite_ratio < 1.0
    bad_low_variance = (std < 5e-6) & (value_range < 1e-4)
    bad_near_black = near_black_ratio > 0.995
    bad_mask = bad_non_finite | bad_low_variance | bad_near_black
    return finite_ratio, std, value_range, near_black_ratio, bad_mask


def _summarize_decoded_metrics(
    finite_ratio: torch.Tensor,
    std: torch.Tensor,
    value_range: torch.Tensor,
    near_black_ratio: torch.Tensor,
    bad_mask: torch.Tensor,
) -> str:
    parts = []
    batch_size = int(finite_ratio.shape[0])
    for i in range(batch_size):
        reasons = []
        if bool(finite_ratio[i] < 1.0):
            reasons.append("non-finite")
        if bool((std[i] < 5e-6) and (value_range[i] < 1e-4)):
            reasons.append("low-variance")
        if bool(near_black_ratio[i] > 0.995):
            reasons.append("near-black")
        reason_text = ",".join(reasons) if reasons else "ok"
        parts.append(
            f"img[{i}]: finite={float(finite_ratio[i]):.6f}, std={float(std[i]):.6f}, "
            f"range={float(value_range[i]):.6f}, black_ratio={float(near_black_ratio[i]):.6f}, "
            f"status={reason_text}"
        )
    bad_count = int(bad_mask.sum().item())
    return f"bad={bad_count}/{batch_size} | " + " | ".join(parts)


def _decode_latents_with_safety(
    pipe: Any, latents: torch.Tensor, generator: torch.Generator, context: str
) -> List[Image.Image]:
    safe_latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0).clamp(-30.0, 30.0)
    if _GLM_DEBUG_ENABLED:
        finite_ratio = float(torch.isfinite(safe_latents).float().mean().item())
        _glm_debug(
            f"{context} latent stats: min={safe_latents.min().item():.6f}, "
            f"max={safe_latents.max().item():.6f}, mean={safe_latents.mean().item():.6f}, finite={finite_ratio:.6f}"
        )

    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.latent_channels, 1, 1)
        .to(safe_latents.device, safe_latents.dtype)
    )
    latents_std = (
        torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.latent_channels, 1, 1)
        .to(safe_latents.device, safe_latents.dtype)
    )

    safe_latents = safe_latents * latents_std + latents_mean

    decoded = pipe.vae.decode(safe_latents.to(pipe.vae.dtype), return_dict=False, generator=generator)[0]
    finite_ratio, std, value_range, near_black_ratio, bad_mask = _decoded_batch_metrics(decoded)
    _glm_debug(
        f"{context} decoded stats(before sanitize): "
        f"{_summarize_decoded_metrics(finite_ratio, std, value_range, near_black_ratio, bad_mask)}, "
        f"dtype={decoded.dtype}"
    )

    # Fallback: fp16/bf16 VAE decode can occasionally collapse to NaN/flat outputs.
    if bool(bad_mask.any()):
        bad_indices = torch.nonzero(bad_mask, as_tuple=False).flatten()
        _glm_debug(f"{context} decode fallback: retrying VAE decode in fp32 for bad indices={bad_indices.tolist()}")
        vae_dtype = pipe.vae.dtype
        try:
            pipe.vae.to(torch.float32)
            bad_latents = safe_latents.index_select(0, bad_indices).to(torch.float32)
            decoded_bad = pipe.vae.decode(bad_latents, return_dict=False, generator=generator)[0]
        finally:
            pipe.vae.to(vae_dtype)

        decoded = decoded.to(torch.float32)
        decoded[bad_indices] = decoded_bad.to(torch.float32)
        finite_ratio, std, value_range, near_black_ratio, bad_mask = _decoded_batch_metrics(decoded)
        _glm_debug(
            f"{context} decoded stats(fp32 fallback): "
            f"{_summarize_decoded_metrics(finite_ratio, std, value_range, near_black_ratio, bad_mask)}, "
            f"dtype={decoded.dtype}"
        )
        if bool(bad_mask.any()):
            bad_indices_after = torch.nonzero(bad_mask, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"{context} 解码异常：fp32 重解码后仍检测到异常图像，indices={bad_indices_after}。"
                "请检查模型权重、显存稳定性或输入条件。"
            )

    decoded = torch.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    return pipe.image_processor.postprocess(decoded, output_type="pil")


def _conditioning_to_text(conditioning: Any) -> str:
    if isinstance(conditioning, dict):
        return str(conditioning.get("text", ""))
    if conditioning is None:
        return ""
    return str(conditioning)


def _is_glm_attention_unpack_error(exc: Exception) -> bool:
    msg = str(exc)
    return "not enough values to unpack" in msg and "expected 2, got 1" in msg


def _run_pipe_with_recovery(pipe: Any, context: str, **kwargs):
    try:
        return pipe(**kwargs)
    except Exception as e:
        if _is_glm_attention_unpack_error(e) and hasattr(pipe, "disable_attention_slicing"):
            _glm_debug(
                f"{context} detected attention unpack mismatch; disabling attention slicing and retrying once"
            )
            try:
                pipe.disable_attention_slicing()
            except Exception as disable_err:
                _glm_debug(
                    f"{context} disable_attention_slicing failed: "
                    f"{type(disable_err).__name__}: {disable_err}"
                )
                raise e
            return pipe(**kwargs)
        raise


def _glm_sample(
    glm_model: Dict[str, Any],
    positive: str,
    seed: int,
    steps: int,
    cfg: float,
    latent: Dict[str, Any],
) -> torch.Tensor:
    if not isinstance(glm_model, dict):
        raise ValueError(f"glm_model 类型错误，期望 dict，实际: {type(glm_model).__name__}")
    if "pipe" not in glm_model or "device" not in glm_model:
        raise ValueError(f"glm_model 缺少关键字段，当前 keys={list(glm_model.keys())}")

    pipe = glm_model["pipe"]
    device = glm_model["device"]
    _patch_processor_apply_chat_template(pipe)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    if not isinstance(latent, dict):
        raise ValueError(f"latent 类型错误，期望 dict，实际: {type(latent).__name__}")

    mode = latent.get("mode", "t2i")
    if mode not in ("t2i", "i2i"):
        raise ValueError(f"latent.mode 非法，期望 t2i 或 i2i，实际: {mode}")

    if "width" not in latent or "height" not in latent:
        raise ValueError(f"latent 缺少 width/height，当前 keys={list(latent.keys())}")
    width = int(latent["width"])
    height = int(latent["height"])
    _validate_size_divisible_by_32(width, height)

    common_args = {
        "prompt": positive,
        "height": height,
        "width": width,
        "num_inference_steps": int(steps),
        "guidance_scale": float(cfg),
        "generator": generator,
    }
    _glm_debug(
        "sample begin: "
        f"mode={mode}, size={width}x{height}, steps={int(steps)}, cfg={float(cfg)}, "
        f"device={device}, dtype={glm_model.get('dtype')}, prompt_len={len(positive or '')}"
    )
    _glm_debug(
        "pipe info: "
        f"class={pipe.__class__.__name__}, patched={getattr(pipe, '_comfy_glm_apply_chat_template_patched', False)}"
    )
    pbar = comfy_utils.ProgressBar(int(steps)) if comfy_utils is not None else None
    fp16_mode = str(glm_model.get("dtype", "")).endswith("float16")

    def _on_step_end(_pipe, step_index, _timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        if fp16_mode and latents is not None:
            finite_mask = torch.isfinite(latents)
            if not bool(finite_mask.all()):
                bad_count = int((~finite_mask).sum().item())
                _glm_debug(f"step={int(step_index)+1}: detected non-finite latents ({bad_count}), applying nan_to_num")
                latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)
            # Guard against fp16 overflow accumulation causing black outputs.
            latents = latents.clamp(-30.0, 30.0)
            callback_kwargs["latents"] = latents

        if pbar is not None:
            pbar.update_absolute(int(step_index) + 1)
        return callback_kwargs

    if mode == "i2i":
        init_image = latent.get("image")
        if init_image is None:
            raise ValueError(f"i2i latent 缺少 image，当前 keys={list(latent.keys())}")
        pil_images = _tensor_to_pil_batch(init_image)
        _glm_debug(f"i2i input image_count={len(pil_images)}")
        try:
            result = _run_pipe_with_recovery(
                pipe=pipe,
                context="i2i",
                **common_args,
                image=pil_images,
                callback_on_step_end=_on_step_end,
                output_type="latent",
            )
        except Exception as e:
            _glm_debug(f"pipe exception (i2i): {type(e).__name__}: {e}")
            _glm_debug(traceback.format_exc())
            raise
        images = _decode_latents_with_safety(pipe, result.images, generator, "i2i")
        _glm_debug_check_images(images, "i2i")
        return _pil_to_tensor_batch(images)

    num_images = int(latent.get("num_images", 1))
    _glm_debug(f"t2i num_images_per_prompt={num_images}")
    try:
        result = _run_pipe_with_recovery(
            pipe=pipe,
            context="t2i",
            **common_args,
            num_images_per_prompt=num_images,
            callback_on_step_end=_on_step_end,
            output_type="latent",
        )
    except Exception as e:
        _glm_debug(f"pipe exception (t2i): {type(e).__name__}: {e}")
        _glm_debug(traceback.format_exc())
        raise
    images = _decode_latents_with_safety(pipe, result.images, generator, "t2i")
    _glm_debug_check_images(images, "t2i")
    return _pil_to_tensor_batch(images)


class GLMImageLoader:
    _NO_MODEL_OPTION = "__NO_MODEL__"

    @classmethod
    def _get_model_choices(cls) -> List[str]:
        choices: List[str] = []
        if folder_paths is None:
            return [cls._NO_MODEL_OPTION]

        try:
            checkpoint_roots = folder_paths.get_folder_paths("checkpoints")
        except Exception:
            checkpoint_roots = []

        for root in checkpoint_roots:
            if not os.path.isdir(root):
                continue
            try:
                entries = sorted(os.listdir(root))
            except Exception:
                continue
            for name in entries:
                full_path = os.path.join(root, name)
                if os.path.isdir(full_path):
                    choices.append(name)

        # 仅保留 checkpoints 目录中的实际文件夹名称
        choices = sorted(set(choices))
        if not choices:
            choices = [cls._NO_MODEL_OPTION]
        return choices

    @classmethod
    def _resolve_model_reference(cls, model_name: str) -> str:
        selected = (model_name or "").strip()
        if not selected or selected == cls._NO_MODEL_OPTION:
            raise ValueError("未找到可用模型目录，请先把模型放到 ComfyUI/models/checkpoints/<目录名>/")

        if folder_paths is None:
            raise ValueError("folder_paths 不可用，无法从 checkpoints 目录解析模型路径")

        try:
            checkpoint_roots = folder_paths.get_folder_paths("checkpoints")
        except Exception as e:
            raise RuntimeError(
                f"读取 checkpoints 路径失败，无法解析模型目录。原始异常：{type(e).__name__}: {e}"
            ) from e

        for root in checkpoint_roots:
            candidate = os.path.join(root, selected)
            if os.path.isdir(candidate):
                return candidate
        raise ValueError(f"在 checkpoints 目录中未找到模型文件夹: {selected}")

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = cls._get_model_choices()
        return {
            "required": {
                "model_name": (model_choices, {"default": model_choices[0]}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "enable_attention_slicing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("GLM_IMAGE_MODEL",)
    RETURN_NAMES = ("glm_model",)
    FUNCTION = "load_model"
    CATEGORY = "GLM-Image/Loaders"

    def load_model(self, model_name, device, dtype, enable_attention_slicing):
        _ensure_dependency()
        try:
            transformers_mod = importlib.import_module("transformers")
            has_glm_image_model = hasattr(transformers_mod, "GlmImageForConditionalGeneration")
            has_glm_image_processor = hasattr(transformers_mod, "GlmImageProcessor")
            if not (has_glm_image_model and has_glm_image_processor):
                transformers_version = getattr(transformers_mod, "__version__", "unknown")
                raise RuntimeError(
                    "当前 transformers 缺少 GLM-Image 所需类："
                    "`GlmImageForConditionalGeneration` / `GlmImageProcessor`。\n"
                    f"当前 transformers 版本：{transformers_version}\n"
                    "请安装包含 GLM-Image 的较新 transformers（通常需要源码版）：\n"
                    "pip install -U git+https://github.com/huggingface/transformers.git"
                )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                "检查 transformers 的 GLM-Image 支持能力失败。\n"
                f"原始异常：{type(e).__name__}: {e}"
            ) from e

        try:
            from diffusers.pipelines.glm_image import GlmImagePipeline
        except Exception as e:
            if "HybridCache" in str(e):
                raise RuntimeError(
                    "GLM-Image pipeline 导入失败：检测到 peft/transformers 版本冲突。\n"
                    "请执行其一：\n"
                    "1) 卸载 peft（不使用 LoRA 时推荐）：`pip uninstall -y peft`\n"
                    "2) 升级 peft 与 transformers 到兼容组合（或直接卸载 peft）\n"
                    f"原始异常：{type(e).__name__}: {e}"
                ) from e
            raise RuntimeError(
                "当前 diffusers 中未找到 GLM-Image pipeline（diffusers.pipelines.glm_image）。\n"
                "请确认已安装支持 GLM-Image 的 diffusers 版本。\n"
                f"原始异常：{type(e).__name__}: {e}"
            ) from e

        model_ref = self._resolve_model_reference(model_name)
        resolved_device = _resolve_device(device)
        resolved_dtype = _resolve_dtype(dtype, resolved_device)
        cache_key = f"{model_ref}|{resolved_device}|{resolved_dtype}|{int(enable_attention_slicing)}"

        with _CACHE_LOCK:
            cached = _PIPELINE_CACHE.get(cache_key)
            if cached is not None:
                _patch_processor_apply_chat_template(cached.get("pipe"))
                return (cached,)

            try:
                pipe = GlmImagePipeline.from_pretrained(
                    model_ref,
                    torch_dtype=resolved_dtype,
                )
            except Exception as e:
                raise RuntimeError(
                    "GLM-Image 模型加载失败（from_pretrained）。\n"
                    f"模型目录: {model_ref}\n"
                    f"推理精度: {resolved_dtype}\n"
                    f"原始异常：{type(e).__name__}: {e}"
                ) from e

            try:
                pipe = pipe.to(resolved_device)
            except Exception as e:
                raise RuntimeError(
                    "GLM-Image 模型迁移设备失败（pipe.to）。\n"
                    f"目标设备: {resolved_device}\n"
                    f"原始异常：{type(e).__name__}: {e}"
                ) from e
            if enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            _patch_processor_apply_chat_template(pipe)

            model_bundle = {
                "pipe": pipe,
                "model_id": model_ref,
                "device": resolved_device,
                "dtype": str(resolved_dtype),
            }
            _PIPELINE_CACHE[cache_key] = model_bundle
            return (model_bundle,)


class GLMImageTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("GLM_IMAGE_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "GLM-Image/Comfy"

    def encode(self, text):
        return ({"text": text},)


class GLMImageEmptyLatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "align_to_32": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("GLM_IMAGE_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "create"
    CATEGORY = "GLM-Image/Comfy"

    def create(self, width, height, batch_size, align_to_32):
        if align_to_32:
            width = _align_to_32(width)
            height = _align_to_32(height)
        return (
            {
                "mode": "t2i",
                "width": int(width),
                "height": int(height),
                "num_images": int(batch_size),
            },
        )


class GLMImageImageToLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_image_size": ("BOOLEAN", {"default": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "align_to_32": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("GLM_IMAGE_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "create"
    CATEGORY = "GLM-Image/Comfy"

    def create(self, image, use_image_size, width, height, align_to_32):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if use_image_size:
            height = int(image.shape[1])
            width = int(image.shape[2])
        if align_to_32:
            width = _align_to_32(width)
            height = _align_to_32(height)
        return (
            {
                "mode": "i2i",
                "image": image,
                "width": int(width),
                "height": int(height),
            },
        )


class GLMImageKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glm_model": ("GLM_IMAGE_MODEL",),
                "positive": ("GLM_IMAGE_CONDITIONING",),
                "latent_image": ("GLM_IMAGE_LATENT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "GLM-Image/Comfy"

    def sample(self, glm_model, positive, latent_image, seed, steps, cfg):
        pos_text = _conditioning_to_text(positive)
        image = _glm_sample(
            glm_model=glm_model,
            positive=pos_text,
            seed=seed,
            steps=steps,
            cfg=cfg,
            latent=latent_image,
        )
        return (image,)


NODE_CLASS_MAPPINGS = {
    "GLMImageLoader": GLMImageLoader,
    "GLMImageTextEncode": GLMImageTextEncode,
    "GLMImageEmptyLatentImage": GLMImageEmptyLatentImage,
    "GLMImageImageToLatent": GLMImageImageToLatent,
    "GLMImageKSampler": GLMImageKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLMImageLoader": "GLM-Image 加载器",
    "GLMImageTextEncode": "GLM-Image Text Encode",
    "GLMImageEmptyLatentImage": "GLM-Image Empty Latent",
    "GLMImageImageToLatent": "GLM-Image Image To Latent",
    "GLMImageKSampler": "GLM-Image KSampler",
}
