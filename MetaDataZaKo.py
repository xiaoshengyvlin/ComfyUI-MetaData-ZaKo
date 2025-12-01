import os
import sys
import time
import numpy as np
from PIL import Image, PngImagePlugin
import folder_paths

# 颜色代码定义
COLORS = {
    "BLUE": "\033[94m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "RESET": "\033[0m"
}

class MetaDataZaKo:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "workflow": ("IMAGE",),
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "我的元信息/ima_"})
            },
            "hidden": {
                "prompt": "PROMPT", 
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "ZaKo"

    def run(self, workflow, image, filename_prefix, prompt=None, unique_id=None):
        # --- 日志函数 (修复版：去除重复，保留穿透力) ---
        def log(msg, color="RESET"):
            text = f"{COLORS.get(color, '')}[ZaKo] {msg}{COLORS['RESET']}\n"
            try:
                # 使用系统底层写入 (os.write)，绕过 Python 打印缓冲区
                # 这既能保证实时显示，又不会像 print 那样被 ComfyUI 拦截
                if sys.platform == 'win32':
                    # Windows 下处理编码，防止乱码
                    os.write(1, text.encode('utf-8', errors='ignore'))
                else:
                    os.write(1, text.encode('utf-8'))
            except Exception:
                # 如果底层写入失败，回退到普通 print
                print(text, end="", flush=True)

        log("------ 开始处理 ------", "BLUE")

        meta_container = PngImagePlugin.PngInfo()
        has_meta = False

        # --- 1. 节点追踪逻辑 ---
        try:
            node_inputs = prompt[unique_id].get('inputs', {}) if prompt and unique_id else {}
            current_link = node_inputs.get('workflow')
            
            if not current_link or not isinstance(current_link, list):
                log("× 错误：workflow 端口未连接！", "RED")
            else:
                current_node_id = current_link[0]
                found_node = None
                
                for _ in range(50):
                    node_obj = prompt.get(current_node_id)
                    if not node_obj: break
                    
                    class_type = node_obj.get('class_type', 'Unknown')
                    log(f"追踪节点 -> ID: {current_node_id} | 类型: {class_type}")

                    if class_type in ["LoadImage", "LoadImageMask", "Load Image"]:
                        found_node = node_obj
                        break
                    
                    if class_type == "Reroute":
                        inputs = node_obj.get('inputs', {})
                        if inputs:
                            prev_link = list(inputs.values())[0]
                            if isinstance(prev_link, list):
                                current_node_id = prev_link[0]
                                continue
                    
                    log(f"× 追踪中断：遇到不支持的节点类型 '{class_type}'", "RED")
                    break

                # --- 2. 元数据提取 ---
                if found_node:
                    img_name = found_node['inputs'].get('image')
                    img_path = folder_paths.get_annotated_filepath(img_name)
                    
                    if img_path and os.path.exists(img_path):
                        log(f"√ 锁定源文件: {img_name}", "BLUE")
                        try:
                            with Image.open(img_path) as src_img:
                                count = 0
                                if getattr(src_img, 'info', None):
                                    for k, v in src_img.info.items():
                                        if k in ["workflow", "prompt"] or "comfy" in k:
                                            meta_container.add_text(k, str(v))
                                            count += 1
                                
                                if count > 0:
                                    has_meta = True
                                    log(f"√ 成功提取 {count} 条元数据信息！", "GREEN")
                                else:
                                    log("× 警告：该图片不包含元数据", "RED")
                        except Exception as e:
                            log(f"× 读取图片文件失败: {e}", "RED")
                    else:
                        log(f"× 警告：文件路径不存在: {img_path}", "RED")

        except Exception as e:
            log(f"× 逻辑异常: {e}", "RED")

        # --- 3. 保存逻辑 ---
        if has_meta:
            log(">>> 正在注入元数据并保存图片...", "BLUE")
        else:
            log("!!! 注意：该图片不包含元数据，注入失败", "RED")

        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, image[0].shape[1], image[0].shape[0])

        results = []
        for batch_number, img_tensor in enumerate(image):
            try:
                i = 255. * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                file_name = f"{filename}_{counter:05}_.png"
                save_path = os.path.join(full_output_folder, file_name)
                
                img.save(save_path, pnginfo=meta_container, compress_level=4)
                
                results.append({
                    "filename": file_name,
                    "subfolder": subfolder,
                    "type": self.type
                })
                counter += 1
            except Exception as save_err:
                log(f"× 保存出错: {save_err}", "RED")

        log("√ 全部完成。图片已保存到 Output 文件夹。", "GREEN")
        
        # 保持这个微小的暂停，确保最后一行字能显示完全
        time.sleep(0.1)

        return {"ui": {"images": results}}

NODE_CLASS_MAPPINGS = { "MetaDataZaKo": MetaDataZaKo }
NODE_DISPLAY_NAME_MAPPINGS = { "MetaDataZaKo": "MetaData Injection (ZaKo)" }
