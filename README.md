# ComfyUI-GLM-Image

将 `zai-org/GLM-Image` 封装为 ComfyUI 可连接节点，支持：

- 文生图（Text-to-Image）
- 图生图（Image-to-Image）
- 与官方示例一致的参数：`prompt`、`height`、`width`、`num_inference_steps`、`guidance_scale`、`generator`

## 安装

在 `ComfyUI` 根目录执行：

```bash
pip install -r custom_nodes/ComfyUI-GLM-Image/requirements.txt
```

> 说明：`GLM-Image` 当前通常依赖 `transformers` 与 `diffusers` 的较新实现，因此这里使用了官方仓库源码安装方式。

## 模型加载方式

- `GLM-Image 加载器` 现在支持 `Checkpoint` 风格下拉选择：
- 将模型目录放到 `ComfyUI/models/checkpoints/<你的目录名>/`
- 在节点的 `model_name` 下拉中选择该目录名即可加载
- 如果目录不兼容 GLM-Image，执行时会正常报错

## 节点列表

- `GLM-Image 加载器`
- `GLM-Image Text Encode`
- `GLM-Image Empty Latent`
- `GLM-Image Image To Latent`
- `GLM-Image KSampler`

## 推荐工作流

### 1) 文生图

1. `GLM-Image 加载器`
2. 一个 `GLM-Image Text Encode`（正向）
3. `GLM-Image Empty Latent`
4. `GLM-Image KSampler`
5. `Save Image`

### 2) 图生图

1. `Load Image`
2. `GLM-Image Text Encode`
3. `GLM-Image Image To Latent`
4. `GLM-Image 加载器`
5. `GLM-Image KSampler`
6. `Save Image`

## 参数建议

- `cfg`：推荐从 `1.2 ~ 3.0` 开始试（GLM-Image 常用较低 guidance）
- `steps`：`30 ~ 60`
- `width/height`：必须是 32 的倍数（插件内已做校验）

## 模型说明（简要）

GLM-Image 采用混合架构：

- 自回归生成器（负责语义结构与知识表达）
- 扩散解码器（负责高保真细节与文字渲染）

同一模型支持 T2I 与 I2I，尤其在“信息密集型画面”和“图中文字准确性”上表现突出。
