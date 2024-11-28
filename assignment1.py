import spacy
import requests
from requests.adapters import HTTPAdapter
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# 设置自定义的User-Agent
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
headers = {
    'User-Agent': user_agent
}

# 定义一个自定义的Wikipedia API类，传入headers
class CustomWikipediaAPI:
    def __init__(self, lang, headers):
        self.lang = lang
        self.headers = headers
        self.base_url = f"https://{self.lang}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.mount('https://', HTTPAdapter(max_retries=3, pool_block=True))

    def page(self, title):
        url = self.base_url + f"?action=query&format=json&titles={title}&prop=extracts&exintro=true"  # 获取简短的摘要
        response = self.session.get(url)
        data = response.json()
        pages = data['query']['pages']
        page = next(iter(pages.values()))
        return page

# 初始化自定义的Wikipedia API
wiki_wiki = CustomWikipediaAPI("en", headers)

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 获取 Wikipedia 页面内容
def get_wikipedia_page_content(page_title):
    
    page = wiki_wiki.page(page_title)  # 获取页面信息
    
    if 'extract' in page:  # 页面包含摘要信息
        content = content = page['extract']
        return content[:500]
        
    else:
        # print(f"Page {page_title} not found.")
        return None

# 获取 BERT 嵌入
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 返回 [CLS] token 的嵌入，表示整个句子的向量
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

# 计算余弦相似度
def cosine_similarity(embedding1, embedding2):
    sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return sim.item()

# 获取候选项的嵌入
def get_candidate_embedding(page_title):
    content = get_wikipedia_page_content(page_title)
    # print("content",content)
    if content:
        return get_bert_embedding(content)
    return None

# # 获取消歧页面的候选项
# def get_disambiguation_candidates(entity):
#     page = wiki_wiki.page(entity + " (disambiguation)")  # 获取消歧页面
#     if 'extract' in page:
#         # 解析消歧页面中的候选项，通常消歧页面会有格式化列表
#         candidates = []
#         content = page['extract']
#         # 假设候选项出现在括号内（如 "Person A", "Company B"）
#         # 这里只是简单的正则提取候选项，具体格式可能需要根据页面内容进行调整
#         for line in content.split("\n"):
#             if "(" in line and ")" in line:
#                 candidate_title = line.split("(")[0].strip()  # 获取候选项标题
#                 candidates.append(candidate_title)
#         return candidates[:5]  # 返回前五个候选项
#     return []

def get_disambiguation_options(entity):
    """
    获取消歧页面中的候选选项。
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": f"{entity} (disambiguation)",
        "prop": "links",
        "format": "json",
        "pllimit": "max"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    pages = data.get("query", {}).get("pages", {})
    disambig_options = []
    
    # 提取消歧页面的候选项
    for page_id, page_data in pages.items():
        if "links" in page_data:
            for link in page_data["links"]:
                disambig_options.append(link["title"])
    
    return disambig_options


# 歧义消除：计算上下文与所有候选项的相似度，并选择最相似的候选项
def disambiguate_entity(context, entity):
    # 获取上下文的 BERT 嵌入
    context_embedding = get_bert_embedding(context)

    best_similarity = -1  # 初始最优相似度为负数
    best_title = None

    # 获取实体的消歧页面候选项
    candidate_titles = get_disambiguation_options(entity)
    # print(candidate_titles)

    if not candidate_titles:
        # print(f"No disambiguation candidates found for {entity}.")
        return None

    for title in candidate_titles:
        candidate_embedding = get_candidate_embedding(title)
        if candidate_embedding is not None:
            similarity = cosine_similarity(context_embedding, candidate_embedding)
            
            # print(f"Cosine similarity between context and {title}: {similarity}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_title = title

    return best_title



# 测试函数
if __name__ == "__main__":

    # 加载预训练的英语模型
    nlp = spacy.load("en_core_web_sm")
    

    context = "Is Managua the capital of Nicaragua?"
    doc = nlp(context)
    for ent in doc.ents:
        entity_name = ent.text
        print(entity_name + " =>", end="")
        best_disambiguation = disambiguate_entity(context, entity_name)
        # print(f"The best disambiguation for the entity is: {best_disambiguation}")
        if best_disambiguation:
            best_link = f"https://en.wikipedia.org/wiki/{best_disambiguation.replace(' ', '_')}"
            print(best_link)
    # entity = "Apple"  # Example entity to disambiguate

    
