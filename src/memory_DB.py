import os
import sqlite3
from datetime import datetime
from langgraph.checkpoint.sqlite import SqliteSaver

# 获取项目根目录 (假设当前文件在 src 目录，根目录为上一级)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")

# 确保 resources 文件夹存在，以便存放多模态数据和 db 文件
os.makedirs(RESOURCES_DIR, exist_ok=True)

# 持久化数据库文件路径
DB_PATH = os.path.join(RESOURCES_DIR, "memory.db")


class AgentMemoryManager:
    """
    Agent 记忆持久化管理器，基于 langgraph-checkpoint-sqlite。
    实现了对多轮对话 Session 的隔离与统一保存。
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # 连接 SQLite 数据库 (check_same_thread=False 允许跨线程/异步操作安全访问)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)
        
        # 初始化检查点相关的数据库表结构
        self.checkpointer.setup()
        
        # 本地会话列表扩展表 (用于记录各个Session的第一句话标签)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS session_metadata (
            thread_id TEXT PRIMARY KEY,
            label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        self.conn.commit()

    def get_checkpointer(self) -> SqliteSaver:
        """
        获取 checkpointer 实例，交给 LangGraph 编译。
        用法: graph.compile(checkpointer=memory_manager.get_checkpointer())
        """
        return self.checkpointer

    @staticmethod
    def generate_new_session_id() -> str:
        """
        以当前时间戳为主体生成唯一会话 ID
        示例: session_20260526_153022
        """
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def create_new_session(self) -> dict:
        """
        开启新会话，生成 ID 并返回供 Agent 执行时的配置。
        返回的 config 可以直接透传给 graph.invoke(..., config=config)
        """
        session_id = self.generate_new_session_id()
        return self.load_session(session_id)

    def load_session(self, session_id: str) -> dict:
        """
        根据给定的会话 ID 加载之前的旧会话
        该配置在传入 graph.invoke 后会自动从 sqlite 中读取该 ID 的上下文
        """
        return {"configurable": {"thread_id": session_id}}

    def list_all_sessions(self) -> list:
        """
        查询并列出目前已保存的所有历史会话详细信息，包括 thread_id 和摘要标签
        """
        cursor = self.conn.cursor()
        try:
            # 联合查询，为了兼容之前没有 metadata 的历史数据，使用左连接
            cursor.execute('''
                SELECT c.thread_id, m.label, m.created_at
                FROM (SELECT DISTINCT thread_id FROM checkpoints) c
                LEFT JOIN session_metadata m ON c.thread_id = m.thread_id
                ORDER BY m.created_at DESC, c.thread_id DESC
            ''')
            rows = cursor.fetchall()
            
            sessions = []
            for row in rows:
                tid = row[0]
                label = row[1] if row[1] else "暂无摘要记录"
                time_str = row[2][:19] if row[2] else "未知时间"  # 取 datetime 部分
                sessions.append({"session_id": tid, "label": label, "created_at": time_str})
            return sessions
        except sqlite3.OperationalError:
            return []

    def save_session_label(self, session_id: str, label: str):
        """
        如果该 session_id 还没有摘要，那么插入其第一句话作为摘要标签
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT label FROM session_metadata WHERE thread_id = ?", (session_id,))
        if not cursor.fetchone():
            _label = label[:20] + "..." if len(label) > 20 else label
            cursor.execute("INSERT INTO session_metadata (thread_id, label) VALUES (?, ?)", (session_id, _label))
            self.conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """
        安全删除指定的会话记录和元数据
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
            cursor.execute("DELETE FROM session_metadata WHERE thread_id = ?", (session_id,))
            self.conn.commit()
            return True
        except Exception as e:
            return False

    def close(self):
        """
        退出机制：安全关闭数据库连接
        """
        if self.conn:
            self.conn.close()


# if __name__ == "__main__":
#     # 快速测试代码
#     manager = AgentMemoryManager()
#     print(f"✅ Memory DB 初始化完毕，路径: {manager.db_path}")
    
#     # 模拟新建会话
#     new_config = manager.create_new_session()
#     print(f"🆕 新创建的 Session Config: {new_config}")
    
#     # 模拟获取历史所有会话
#     sessions = manager.list_all_sessions()
#     print(f"📜 当前的历史 Session 列表: {sessions}")
    
#     manager.close()