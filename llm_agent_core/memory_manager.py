import chromadb
import numpy as np
import json
from typing import List, Dict

import re
import os
from typing import Dict, Any


# the semantica segementation  "../scence_data/seg_map/"
def _parse_filename_to_ned(filename: str) -> Dict[str, float]:
    """从文件名 'X=...Y=...Z=...' 中解析出 NED 坐标。"""
    match = re.search(r'X=([-+]?\d*\.?\d+)Y=([-+]?\d*\.?\d+)Z=([-+]?\d*\.?\d+)', filename)
    if match:
        return {
            "x": float(match.group(1)),
            "y": float(match.group(2)),
            "z_down": float(match.group(3))
        }
    return None

def _create_semantic_description(data: Dict[str, str], ned_coords: Dict[str, float]) -> str:
    """将语义信息和位置组合成一个自然语言描述。"""
    
    # 构建语义描述
    description = f"一个类型为 '{data['type']}' 的物体/区域被检测到。特征描述："
    features = [f"{k}: {v}" for k, v in data.items() if k not in ['type', 'filename'] and v != 'test']
    description += ", ".join(features)
    
    # 添加位置信息
    description += (f" 位于无人机拍摄点 (NED 坐标: X={ned_coords['x']:.2f}, Y={ned_coords['y']:.2f}, "
                    f"海拔高度: {-ned_coords['z_down']:.2f} 米)。")
    
    return description



# --- 核心记忆管理类 ---
class MemoryManager:
    """
    Agent 的长期记忆 (LTM) 管理器，使用 ChromaDB 实现 RAG 机制。
    现在支持基于场景 ID 的持久化存储。
    """

    SEG_MAP_BASE_DIR = 'scene_data/seg_map'
    CHROMA_DB_BASE_DIR = "./chroma_dbs"
    
    def __init__(self, scene_name: str, load_static_map: bool = False, embedding_dim: int = 384):
        """
        初始化 MemoryManager。
        
        参数:
            scene_name (str): 当前场景的唯一标识符。
            load_static_map (bool): 是否在初始化时导入静态地图数据。
        """
        self.scene_name = scene_name
        self.load_static_map = load_static_map
        self.embedding_dim = embedding_dim
        
        # 🌟 关键修正 1: 定义 collection_name
        self.collection_name = f"uav_memory_{scene_name}" 
        
        # 1. 初始化 ChromaDB 客户端 (使用持久化模式)
        db_path = os.path.join(self.CHROMA_DB_BASE_DIR, scene_name)
        os.makedirs(db_path, exist_ok=True) 
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 🌟 关键修正 2: 使用 collection_name 创建集合
        # 注意：在真实的 ChromaDB 集成中，name 参数是 get_or_create_collection 必需的
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name  # <--- 使用定义的名称
            # embedding_function=... # 真实项目需指定
        )
        
        print(f"记忆管理器初始化成功，场景: {self.scene_name}，集合: {self.collection_name}")
        print(f"静态地图加载状态: {'启用' if self.load_static_map else '禁用'}。") 

    
    def check_and_initialize_scene_data(self) -> str:
        """
        检查场景静态数据是否已导入。现在受 load_static_map 标志控制。
        """
        
        # 如果 load_static_map 为 False，则直接跳过导入
        if not self.load_static_map:
            return f"OBSERVATION: 静态地图数据加载已禁用 (load_static_map=False)。"

        # 如果 load_static_map 为 True，则继续检查集合是否有数据
        
        jsonl_filename = f"{self.scene_name}_seg_map.jsonl" # 假设文件命名约定
        jsonl_filepath = os.path.join(self.SEG_MAP_BASE_DIR, jsonl_filename)
        
        if self.collection.count() > 0:
            # 注意：即使 load_static_map=True，如果数据库中已有数据，我们仍然跳过导入，使用已有的持久化数据
            print(f"✅ 场景 '{self.scene_name}' 记忆库已包含 {self.collection.count()} 条记录，跳过静态导入。")
            return f"OBSERVATION: 场景 '{self.scene_name}' 记忆库已初始化（包含静态数据）。"
        else:
            print(f"⚠ 场景 '{self.scene_name}' 记忆库为空，load_static_map=True，正在尝试从 {jsonl_filepath} 导入...")
            # 导入静态数据
            return self.import_semantic_jsonl_data(jsonl_filepath)
            

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
    

    def import_semantic_jsonl_data(self, jsonl_filepath: str) -> List[str]:
        """
        从 JSONL 文件中导入语义分割数据到 RAG 向量数据库。
        """
        imported_ids = []
        documents_to_add = []
        metadatas_to_add = []
        
        try:
            with open(jsonl_filepath, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        # 1. 解析坐标
                        ned_coords = _parse_filename_to_ned(data['filename'])
                        if not ned_coords:
                            continue
                        
                        # 2. 创建语义文档和元数据
                        semantic_doc = _create_semantic_description(data, ned_coords)
                        
                        # 3. 构造元数据 (将所有原始字段也存入，便于检索)
                        metadata = {**data, **ned_coords, 'source': 'semantic_segmentation', 'status': 'detected'}
                        
                        # 4. 存储
                        documents_to_add.append(semantic_doc)
                        metadatas_to_add.append(metadata)
                        # 使用 UUID 或一个基于内容的哈希值作为 ID
                        doc_id = f"sem_{hash(semantic_doc)}".replace('-', '')
                        imported_ids.append(doc_id)
                        
                    except json.JSONDecodeError:
                        print(f"警告: 跳过无效的 JSONL 行: {line.strip()[:50]}...")
                        continue

            if documents_to_add:
                # 假设您使用 ChromaDB，并有一个 self.collection 对象
                # self.collection.add(
                #     documents=documents_to_add,
                #     metadatas=metadatas_to_add,
                #     ids=imported_ids
                # )
                print(f"✅ 成功导入 {len(documents_to_add)} 条语义分割记录到记忆库。")
                return imported_ids
            
            return []

        except FileNotFoundError:
            print(f"❌ 错误: 找不到文件 {jsonl_filepath}")
            return []

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