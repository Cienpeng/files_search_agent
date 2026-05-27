import sys
import os
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 加入路径以便引用其他目录模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.tool_read_full_document import read_local_file_content
from tools.tool_semantic_search import search_local_knowledge
from tools.tool_image_search import search_local_image_by_content
from tools.tool_file_search import search_file_in_d_drive
from load_model import InitModel
from memory_DB import AgentMemoryManager

# === 1. 定义 Graph 的状态 (State) ===
# langgraph 要求我们在节点中传递的上下文类型
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 对话流消息，使用 add_messages 实现上下文追加

# === 2. 初始化大模型与绑定 Tools ===
def init_agent_llm():
    # 利用你写好的 load_model.py 初始化指定的大模型
    llm_initializer = InitModel(model_name="deepseek-chat", model_provider="openai", base_url=None, api_key=None)
    llm = llm_initializer.create_chat_model()
    
    # 把我们的自定义 Tool 注册进来
    tools = [
        search_file_in_d_drive,           # 【工具1】纯找文件名
        search_local_knowledge,           # 【工具2】查找知识（文字相关：涵盖从PDF或图片OCR提炼出来的字）
        search_local_image_by_content,    # 【工具3】找画面内容（CN-CLIP相关）
        read_local_file_content           # 【工具4】精确通读某PDF的指定页或提取图片OCR文字
    ]
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tools

# === 3. 定义执行节点 ===
def build_agent_graph(checkpointer=None):
    llm_with_tools, tools = init_agent_llm()
    
    # 核心推理节点
    def chatbot_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
        
    # 构建图 Workflow
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("chatbot", chatbot_node)
    
    # LangGraph 内置的 ToolNode 支持自动处理多工具参数解析
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # 定义跳转逻辑
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # 所有 tool 执行完后重新流转回模型进行润色回答
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    
    # 编译并注入记忆检查点
    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph

def chat_run():
    """本地简易测试入口"""
    # 1. 实例化 SQLite 记忆管理器
    memory_mgr = AgentMemoryManager()
    
    # 2. 编译带有 Checkpointer 记忆存储功能的 Agent 图
    graph = build_agent_graph(checkpointer=memory_mgr.get_checkpointer())
    
    print("🤖 Agent已启动，支持功能：\n1.全局D盘纯文件名搜索\n2.语义搜索(找文档内容/图片上的人话)\n3.找照片(搜图画面特征)\n4.通读特定PDF\n输入 'exit' 退出。")
    print("-" * 50)
    
    # 打印所有的Session历史
    sessions = memory_mgr.list_all_sessions()
    if sessions:
        print("📜 历史会话列表：")
        for idx, s in enumerate(sessions):
            print(f"  [{idx+1}] {s['session_id']} | 📝摘要: {s['label']}")
        print(" 💡 提示: 输入数字序号直接恢复，输入 'del 数字' 删除会话 (例: del 1)，直接回车将开启新会话。")
    else:
        print("📜 暂无历史会话记录。")
    
    # 3. 询问用户是否载入之前的会话
    session_id = None
    while True:
        choice = input("\n📝 请输入选项 (数字恢复/del 删除/直接回车全新开启): ").strip()
        if not choice:
            session_id = memory_mgr.generate_new_session_id()
            print(f"[*] 已创建新会话上下文: {session_id} (数据保存在 memory.db)")
            break
        elif choice.lower().startswith("del "):
            del_idx_str = choice[4:].strip()
            if del_idx_str.isdigit() and 1 <= int(del_idx_str) <= len(sessions):
                del_session = sessions[int(del_idx_str) - 1]["session_id"]
                if memory_mgr.delete_session(del_session):
                    print(f"[*] 已成功删除会话: {del_session}")
                    # 重新刷新当前列表状态
                    sessions = memory_mgr.list_all_sessions()
                else:
                    print(f"❌ 删除失败: {del_session}")
            else:
                print("⚠️ 输入格式错误或序号超出范围，请重试。")
        elif choice.isdigit() and 1 <= int(choice) <= len(sessions):
            session_id = sessions[int(choice) - 1]["session_id"]
            print(f"[*] 正在恢复历史会话进度: {session_id}")
            break
        else:
            # 兼容以前直接输入 session_id 字符串的恢复手段
            session_id = choice
            print(f"[*] 正在尝试手动恢复会话: {session_id}")
            break
        
    # 定义当前的持久化配置
    config = {"configurable": {"thread_id": session_id}}
    
    while True:
        try:
            user_input = input("User >> ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
                
            # 保存第一句话为标签
            memory_mgr.save_session_label(session_id, user_input)
            
            # 使用 langgraph 的 stream 方法观察 Agent 的推理步骤
            # 注意：传入 user_input 时，只需加入当前这一句，
            # 历史记录会自动由 checkpointer 从 SQLite 库里提取拼接！
            events = graph.stream(
                {"messages": [("user", user_input)]}, 
                config=config,
                stream_mode="values"
            )
            for event in events:
                if "messages" in event:
                    # 总是取最后一条消息看动作状态
                    latest_msg = event["messages"][-1]
                    if latest_msg.type == "ai" and not latest_msg.tool_calls:
                        # 最终的AI直接答复
                        print(f"AI >> {latest_msg.content}\n")
                    elif latest_msg.type == "ai" and latest_msg.tool_calls:
                        # 指示它要调用的工具
                        print(f"🛠️ [Agent Action] AI打算调用工具 -> {[t['name'] for t in latest_msg.tool_calls]}")
                    elif latest_msg.type == "tool":
                        # 工具的原始底层返回
                        print(f"⚙️ [Tool Result] -> (由于文本过长不打印在终端，移交回模型层...)")
        except Exception as e:
            print(f"❌ 运行发生错误: {str(e)}")

if __name__ == "__main__":
    chat_run()
