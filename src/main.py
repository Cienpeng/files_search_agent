import sys
import os

# 确保能正确引入相对目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.agent_graph import chat_run

def main():
    print("====== 欢迎使用 Flies Search Agent 多模态检索助手 ======")
    # 直接调用 agent_graph 里封装好的对话入口
    chat_run()

if __name__ == "__main__":
    main()
