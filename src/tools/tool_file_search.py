import json
from langchain_core.tools import tool
import sys
import os

# 引入 files_index 模块中的 file_index 类
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from files_index import file_index

@tool
def search_file_in_d_drive(filename_keyword: str) -> str:
    """
    仅当用户明确要求查找文件路径、文件位置、某文件在哪里，或者知识库检索未命中后需要按文件名兜底搜索时，调用此工具。
    如果用户同时询问文件内容、课程安排、文档里的信息、图片里的文字等内容问题，应优先使用 search_local_knowledge 或 read_local_file_content，而不是先调用本工具。
    出于安全规范，该工具通过底层命令限制，【仅允许】且【强制只】在本地的 D 盘 (D:\\) 内进行搜索，禁止越权搜索其它盘符。
    
    Args:
        filename_keyword: 需要搜索的文件名关键词（不需要带通配符），例如 "test.pdf", "会议记录"。
    """
    try:
        print(f"[*] Agent 正在调用快速 NTFS 搜索插件搜索: D:/*{filename_keyword}*")
        
        # 实例化 file_index
        indexer = file_index()
        # 调用搜索，盘符 "D:/", 模式会自动将关键字加上通配符
        pattern = f"*{filename_keyword}*"
        results = indexer.search("D:/", pattern, case_sensitive=False)
        
        if results:
            total = len(results)
            max_display = 15 # 只返回前几个，避免撑爆 Token
            
            output_lines = []
            for item in results[:max_display]:
                
                # 根据 flies_index.py 示例返回的是 dict 
                output_lines.append(json.dumps(item, ensure_ascii=False))
                
            msg = f"✅ 成功在 D 盘找到了 {total} 个与 '{filename_keyword}' 相关的文件。\n"
            msg += "为您截取前几个搜索结果如下：\n"
            msg += "\n".join(output_lines)
            if total > max_display:
                msg += f"\n\n...(省略其它 {total - max_display} 个结果)"
            return msg
        else:
            return f"❌ 未能在 D 盘的任何目录下找到包含关键字 '{filename_keyword}' 的文件。"
            
    except Exception as e:
        return f"❌ 检索执行失败，错误原因：{str(e)}"
