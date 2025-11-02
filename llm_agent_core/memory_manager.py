import chromadb
import numpy as np
import json
from typing import List, Dict

# --- 核心记忆管理类 ---

class MemoryManager:
    """
    Agent 的长期记忆 (LTM) 管理器，使用 ChromaDB 实现 RAG 机制。
    用于存储搜索线索、已探索区域等，并支持语义搜索召回。
    """
    
    def __init__(self, collection_name: str = "uav_search_memory", embedding_dim: int = 384):
        """
        初始化 ChromaDB 客户端和向量集合。
        我们使用 in-memory 模式进行 PoC 快速验证。
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # 1. 初始化 ChromaDB 客户端 (in-memory 模式)
        self.client = chromadb.Client() 
        
        # 2. 创建或获取向量集合
        # 注意：ChromaDB 需要一个 Embedding Function，这里我们用一个模拟函数
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        print(f"记忆管理器初始化成功，集合: {self.collection_name}，维度: {self.embedding_dim}")

    def _get_embedding(self, text: str) -> List[float]:
        """
        【模拟函数】将文本转换为向量嵌入。
        在真实项目中，这里应调用 OpenAI, Cohere, 或 HuggingFace 等嵌入模型 API。
        """
        # PoC 模拟：返回一个归一化的随机向量
        vector = np.random.rand(self.embedding_dim)
        return (vector / np.linalg.norm(vector)).tolist()

    # --- LLM-Agent 工具 1: 存储/更新记忆 ---
    
    def update_search_map(self, coordinates: Dict[str, float], status: str, description: str) -> str:
        """
        【LLM-Agent 工具】Agent 发现新线索或完成搜索区域时调用。
        
        参数:
            coordinates (dict): 发现点的 GPS 坐标 (latitude, longitude, altitude)。
            status (str): 记忆的状态 (例如: "线索", "已搜索", "禁飞区")。
            description (str): 详细描述。
            
        返回:
            str: 记忆更新结果的观察报告。
        """
        # 构建用于嵌入和存储的文本
        memory_text = f"在坐标 (Lat:{coordinates['latitude']:.5f}, Lon:{coordinates['longitude']:.5f}) 处：状态='{status}'，描述='{description}'。"
        
        embedding = self._get_embedding(memory_text)
        
        # 为每个点生成一个唯一的 ID
        item_id = str(len(self.collection.get()['ids']) + 1)
        
        self.collection.add(
            embeddings=[embedding],
            documents=[memory_text],
            metadatas=[{"status": status, "coordinates": json.dumps(coordinates), "description": description}],
            ids=[item_id]
        )
        
        return f"OBSERVATION: 长期记忆更新成功。新增记忆 ID: {item_id}，内容：'{memory_text}'。"

    # --- LLM-Agent 工具 2: 召回历史线索 (RAG) ---

    def retrieve_historical_clues(self, query: str, top_k: int = 3) -> str:
        """
        【LLM-Agent 工具】根据 Agent 的语义查询，召回最相关的历史线索。
        
        参数:
            query (str): LLM Agent 提出的自然语言查询 (例如: "我最后在哪里看到了红色的东西?")。
            top_k (int): 召回最相关的记忆数量。
            
        返回:
            str: 召回的历史线索，以结构化文本形式返回给 LLM。
        """
        
        query_vector = self._get_embedding(query)
        
        # 使用 ChromaDB 进行相似性搜索
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        clues_list = []
        if results and results['documents'] and results['documents'][0]:
            
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                clues_list.append(
                    f"[距离 {dist:.4f}]: {doc}"
                )
        
        if not clues_list:
            return "OBSERVATION: 长期记忆中没有发现与查询相关的历史线索。"
        
        # 将召回结果格式化，以便 LLM 容易解析和整合到 Thought 中
        return "OBSERVATION: 召回的历史线索如下：\n" + "\n".join(clues_list)

# --- 示例用法 ---

if __name__ == "__main__":
    memory_manager = MemoryManager()

    # 1. 测试记忆存储
    test_coords_1 = {"latitude": 47.641, "longitude": -122.140, "altitude": 30.0}
    test_coords_2 = {"latitude": 47.642, "longitude": -122.139, "altitude": 35.0}

    # 记忆 1: 发现线索
    result_1 = memory_manager.update_search_map(
        test_coords_1, "线索", "发现一个红色的背包，但无人机无法靠近。"
    )
    print(result_1)

    # 记忆 2: 探索完成
    result_2 = memory_manager.update_search_map(
        test_coords_2, "已搜索", "完成了该区域的网格搜索，没有其他发现。"
    )
    print(result_2)

    # 2. 测试 RAG 召回
    print("\n--- Agent 询问：红色的线索在哪里？ ---")
    retrieval_query_1 = "红色的线索在哪里？"
    retrieval_result_1 = memory_manager.retrieve_historical_clues(retrieval_query_1)
    print(retrieval_result_1)
    
    print("\n--- Agent 询问：我搜完了哪些地方？ ---")
    retrieval_query_2 = "哪些地方已经完成了搜索？"
    retrieval_result_2 = memory_manager.retrieve_historical_clues(retrieval_query_2)
    print(retrieval_result_2)