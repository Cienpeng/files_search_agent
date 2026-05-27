# 为文件生成索引树import ctypes, json
import ctypes
import json
import os

# 获取脚本所在目录，然后获取DLL的完整路径


# 盘符，通配符，是否区分大小写（0表示不区分，1表示区分）
# ptr = dll.NtfsSearchJson("D:/论文", "*.pdf", 0)
# json_str = ctypes.wstring_at(ptr)
# dll.NtfsFreeJson(ptr)

# data = json.loads(json_str)
# print(data)

class file_index:
    _dll_instance = None  # 类级缓存，避免重复加载DLL
    
    def __init__(self):
        if file_index._dll_instance is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(script_dir, "../lib/NTFS-Search.dll")
            if not os.path.exists(dll_path):
                raise FileNotFoundError(f"DLL not found at {dll_path}")
            
            dll = ctypes.WinDLL(dll_path)
            dll.NtfsSearchJson.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int]
            dll.NtfsSearchJson.restype = ctypes.c_void_p
            dll.NtfsFreeJson.argtypes = [ctypes.c_void_p]
            dll.NtfsFreeJson.restype = None
            file_index._dll_instance = dll
        
        self.dll = file_index._dll_instance
    
    def search(self, drive_letter: str, pattern: str, case_sensitive: bool = False) -> list[dict]:
        '''搜索指定盘符下符合模式的文件，返回包含文件路径和元数据的列表
        Args:
            drive_letter: 盘符，如 "D:/"
            pattern: 搜索模式，如 "*.pdf"
            case_sensitive: 是否区分大小写，默认为 False
        '''
        try:
            ptr = self.dll.NtfsSearchJson(drive_letter, pattern, int(case_sensitive))
            if not ptr:
                return []
            
            json_str = ctypes.wstring_at(ptr)
            self.dll.NtfsFreeJson(ptr)
            return json.loads(json_str) if json_str else []
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"Search failed: {e}") from e