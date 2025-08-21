# from fastapi import FastAPI, APIRouter
# from pydantic import BaseModel
# from sqlalchemy import create_engine, text
# from langchain_community.vectorstores import FAISS
# from langchain.docstore.document import Document
# from langchain.chains import RetrievalQA
# from langchain_core.runnables import RunnableSequence
# from langchain.prompts import PromptTemplate
# from langchain_cohere.embeddings import CohereEmbeddings
# from langchain_cohere.chat_models import ChatCohere
# import cohere
# import os
# from dotenv import load_dotenv
# import base64
# import re
# import json
# from starlette.middleware.cors import CORSMiddleware
# import time
# from functools import wraps
# from requests.exceptions import HTTPError
#
# # --- LOAD ENVIRONMENT VARIABLES ---
# load_dotenv()
# cohere_api_key = os.getenv("COHERE_API_KEY")
# if not cohere_api_key:
#     raise ValueError("COHERE_API_KEY not found in environment variables. Please check .env file.")
#
# # --- DATABASE CONFIGURATION ---
# DB_PARAMS = {
#     "dbname": os.getenv("DB_NAME", "fashion-shop-3"),
#     "user": os.getenv("DB_USER", "postgres"),
#     "password": os.getenv("DB_PASSWORD", "123456"),
#     "host": os.getenv("DB_HOST", "localhost"),
#     "port": os.getenv("DB_PORT", "5432")
# }
# engine = create_engine(
#     f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}")
#
# # --- FETCH CATEGORIES FROM DATABASE ---
# def fetch_categories():
#     query = "SELECT id, name FROM public.categories"
#     try:
#         with engine.connect() as conn:
#             result = conn.execute(text(query))
#             categories = {row.id: row.name.lower() for row in result}
#             if not categories:
#                 raise ValueError("No categories found in database. Please ensure the 'categories' table is populated.")
#             print(f"Fetched {len(categories)} categories from database: {categories}")
#             return categories
#     except Exception as e:
#         print(f"Error fetching categories from database: {e}")
#         raise ValueError("Failed to fetch categories from database. Application cannot start without valid categories.")
#
# # Load categories at startup and validate
# CATEGORY_MAPPING = fetch_categories()
# REQUIRED_CATEGORIES = {"áo", "quần", "giày"}
# if not all(cat in CATEGORY_MAPPING.values() for cat in REQUIRED_CATEGORIES):
#     missing = REQUIRED_CATEGORIES - set(CATEGORY_MAPPING.values())
#     raise ValueError(f"Missing required categories {missing} in CATEGORY_MAPPING: {CATEGORY_MAPPING}")
#
# # --- VALID GENDERS ---
# VALID_GENDERS = ['MALE', 'FEMALE']
# UNISEX_LABEL = "UNISEX"
#
# # --- RATE LIMITER DECORATOR ---
# def rate_limit(max_per_minute):
#     min_interval = 60.0 / max_per_minute
#     last_called = [0.0]
#
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             elapsed = time.time() - last_called[0]
#             time_to_wait = min_interval - elapsed
#             if time_to_wait > 0:
#                 time.sleep(time_to_wait)
#             result = func(*args, **kwargs)
#             last_called[0] = time.time()
#             return result
#         return wrapper
#     return decorator
#
# # --- FETCH PRODUCTS FROM POSTGRES WITH FILTERS ---
# def fetch_product_descriptions(category=None, gender=None, max_price=None):
#     query = "SELECT id, name, description, gender, price, category_id, thumbnail FROM public.products WHERE status = 't'"
#     params = {}
#     if category:
#         category_id = next((k for k, v in CATEGORY_MAPPING.items() if v == category.lower()), None)
#         if category_id is None:
#             print(f"Error: Category '{category}' not found in CATEGORY_MAPPING: {CATEGORY_MAPPING}")
#             return []
#         query += " AND category_id = :category_id"
#         params['category_id'] = category_id
#     if gender:
#         if gender.upper() in VALID_GENDERS or gender.upper() == UNISEX_LABEL:
#             query += " AND gender = :gender"
#             params['gender'] = gender.upper()
#         else:
#             print(f"Warning: Invalid gender '{gender}', skipping gender filter.")
#
#     if max_price:
#         query += " AND price <= :max_price"
#         params['max_price'] = float(max_price)
#
#     try:
#         with engine.connect() as conn:
#             print(f"Executing SQL: {query}, Parameters: {params}")
#             result = conn.execute(text(query), params)
#             products = []
#             for row in result:
#                 thumb_url = "https://via.placeholder.com/120"
#                 if row.thumbnail:
#                     if isinstance(row.thumbnail, (bytes, bytearray)):
#                         b64_str = base64.b64encode(row.thumbnail).decode("utf-8")
#                         thumb_url = f"data:image/jpeg;base64,{b64_str}"
#                     elif isinstance(row.thumbnail, str):
#                         thumb_url = row.thumbnail
#
#                 products.append({
#                     "id": str(row.id),
#                     "name": row.name,
#                     "description": row.description,
#                     "gender": row.gender,
#                     "price": float(row.price),
#                     "thumbnail": thumb_url,
#                     "category": CATEGORY_MAPPING.get(row.category_id, "khác")
#                 })
#             print(f"Fetched {len(products)} products for category: {category}, gender: {gender}, max_price: {max_price}")
#             return products
#     except Exception as e:
#         print(f"Error fetching data from database: {e}")
#         return []
#
# # --- PARSE USER QUERY USING COHERE LLM ---
# llm = ChatCohere(cohere_api_key=cohere_api_key)
#
# parse_prompt_template = PromptTemplate(
#     input_variables=["query"],
#     template="""
#     Phân tích câu hỏi của người dùng bằng tiếng Việt một cách chính xác:
#     - Nếu chứa từ như 'đồ', 'bộ đồ', 'set', 'set đồ', 'outfit' hoặc không chỉ định danh mục cụ thể (áo, quần, giày), thì intent là 'gợi ý set đồ' và phân tích thành 3 phần danh mục (áo, quần, giày) với mô tả ngắn gọn cho từng phần.
#     - Nếu chỉ định chính xác 1 danh mục (ví dụ: áo, quần, giày, váy, phụ kiện), thì intent là 'gợi ý sản phẩm', category là danh mục đó, không lẫn lộn các danh mục khác.
#     - Nếu câu hỏi không chứa từ khóa liên quan đến sản phẩm (như áo, quần, giày, set, đồ, giá, v.v.) và mang tính trò chuyện (ví dụ: 'chào', 'hôm nay thế nào', 'kể chuyện'), thì intent là 'trò chuyện'.
#     - Trích xuất giới tính (nam -> 'MALE', nữ -> 'FEMALE', hoặc '' nếu không xác định).
#     - Trích xuất giá tối đa (nếu có, chuyển về số, ví dụ: 2 triệu -> 2000000).
#     - Trả về **chỉ** JSON hợp lệ, không sử dụng markdown wrapper (```json hoặc ```):
#     {{
#       "intent": "gợi ý set đồ" hoặc "gợi ý sản phẩm" hoặc "trò chuyện",
#       "category": "danh mục nếu không phải set hoặc trò chuyện, hoặc null",
#       "gender": "MALE" hoặc "FEMALE" hoặc "",
#       "max_price": null hoặc số,
#       "parts": [] nếu set đồ thì là list 3 phần mô tả, ví dụ ["Áo: mô tả", "Quần: mô tả", "Giày: mô tả"], nếu không phải set thì []
#     }}
#     Câu hỏi: {query}
#     """
# )
#
# @rate_limit(max_per_minute=10)
# def call_cohere_api(chain, input_data):
#     return chain.invoke(input_data)
#
# def parse_user_query(query):
#     parse_chain = RunnableSequence(parse_prompt_template | llm)
#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             result = call_cohere_api(parse_chain, {"query": query})
#             raw_output = result.content.strip()
#             print(f"Raw LLM output (attempt {attempt + 1}): {raw_output}")
#             raw_output = raw_output.encode('utf-8').decode('utf-8-sig')
#             raw_output = re.sub(r'^```json\n|\n```$', '', raw_output).strip()
#             print(f"Cleaned output (attempt {attempt + 1}): {raw_output}")
#             if not raw_output or not (raw_output.startswith('{') and raw_output.endswith('}')):
#                 print(f"Error: LLM output is not valid JSON (attempt {attempt + 1})")
#                 continue
#             parsed = json.loads(raw_output)
#             print(f"Parsed JSON: {parsed}")
#             gender = parsed.get("gender") or ""
#             if gender and gender.upper() not in VALID_GENDERS:
#                 if gender.upper() == UNISEX_LABEL:
#                     pass
#                 else:
#                     print(f"Warning: Invalid gender '{gender}', defaulting to ''")
#                     gender = ""
#             parts = parsed.get("parts", [])[:3]  # Ensure only 3 parts
#             if len(parsed.get("parts", [])) > 3:
#                 print(f"Warning: LLM returned {len(parsed.get('parts', []))} parts, truncating to 3.")
#             return (
#                 parsed.get("intent"),
#                 parsed.get("category"),
#                 gender,
#                 parsed.get("max_price"),
#                 parts
#             )
#         except json.JSONDecodeError as e:
#             print(f"JSON parse error in parse_user_query (attempt {attempt + 1}): {e}")
#             print(f"Raw output causing error: {repr(raw_output)}")
#             continue
#         except Exception as e:
#             print(f"Error parsing query (attempt {attempt + 1}): {e}")
#             continue
#     print("Error: Failed to parse JSON after all attempts")
#     if "áo" in query.lower() and "nam" in query.lower():
#         return "gợi ý sản phẩm", "áo", "MALE", None, []
#     return None, None, None, None, []
#
# # --- EMBEDDING WITH COHERE ---
# co = cohere.Client(cohere_api_key)
# embedding_model = CohereEmbeddings(
#     cohere_api_key=cohere_api_key,
#     model="embed-multilingual-v3.0"
# )
#
# # --- LOAD DATA AND CREATE INITIAL VECTOR STORE ---
# products = fetch_product_descriptions()
# if not products:
#     print("No products found in database.")
# docs = [Document(page_content=f"{p['name']}: {p['description']}", metadata={
#     "id": p["id"],
#     "name": p["name"],
#     "gender": p["gender"],
#     "price": p["price"],
#     "thumbnail": p["thumbnail"],
#     "category": p["category"]
# }) for p in products]
# db = FAISS.from_documents(docs, embedding_model)
#
# # --- PROMPT FOR RETRIEVAL QA (PRODUCT MATCHING) ---
# qa_prompt_template = PromptTemplate(
#     input_variables=["question", "context"],
#     template="""
#     Dựa trên mô tả: {question}
#     Hãy gợi ý sản phẩm phù hợp từ context, đảm bảo danh mục đúng (ví dụ: áo, quần, giày, váy, phụ kiện).
#     Context: {context}
#     Trả về mô tả sản phẩm phù hợp dưới dạng văn bản tự nhiên bằng tiếng Việt, không sử dụng JSON hoặc markdown. Ví dụ: "Áo sơ mi nữ H&M, chất liệu cotton pha lụa, màu trắng kem, phù hợp cho công sở."
#     """
# )
#
# # --- HELPER FUNCTION TO FORMAT ANSWER FIELD AS USER-FRIENDLY TEXT ---
# def format_answer_to_text(intent, parts, product_suggestions=None, single_category_answer=None, query=""):
#     if intent == "gợi ý set đồ" and parts:
#         header = "Dựa trên yêu cầu của bạn, đây là gợi ý set đồ:" if "set đồ" in query.lower() else "Dựa trên yêu cầu của bạn, đây là gợi ý:"
#         response_text = f"{header}\n\n"
#         response_text += "Set đồ gợi ý:\n"
#         for part in parts:
#             response_text += f"- {part}\n"
#         response_text += "\n"
#         if product_suggestions:
#             response_text += "Chi tiết sản phẩm:\n"
#             for part, suggestion in product_suggestions:
#                 response_text += f"- {part}\n"
#                 response_text += f"  {suggestion}\n"
#         return response_text.strip()
#     elif intent == "gợi ý sản phẩm" and single_category_answer:
#         header = f"Dựa trên yêu cầu của bạn, đây là gợi ý sản phẩm cho {single_category_answer.get('category', '')}:"
#         response_text = f"{header}\n\n"
#         response_text += single_category_answer.get("answer", "Không tìm thấy sản phẩm phù hợp.")
#         return response_text.strip()
#     return "Không thể xử lý yêu cầu. Vui lòng thử lại."
#
# router = APIRouter()
#
# class Query(BaseModel):
#     question: str
#
# @router.post("/chat")
# def chat(query: Query):
#     try:
#         intent, category, gender, max_price, parts = parse_user_query(query.question)
#         print(f"Parsed intent: {intent}, category: {category}, gender: {gender}, max_price: {max_price}, parts: {parts}")
#         sources_list = []
#
#         # -----------------------
#         # TRÒ CHUYỆN PATH
#         # -----------------------
#         if intent == "trò chuyện":
#             # Gọi LLM trực tiếp để trả lời tự nhiên
#             chat_prompt = PromptTemplate(
#                 input_variables=["question"],
#                 template="""
#                 Trả lời câu hỏi của người dùng bằng tiếng Việt một cách tự nhiên, thân thiện, như một cuộc trò chuyện thông thường. Không đề cập đến sản phẩm, gợi ý mua sắm, hoặc bất kỳ nội dung liên quan đến cửa hàng trừ khi được yêu cầu rõ ràng.
#                 Câu hỏi: {question}
#                 """
#             )
#             chat_chain = RunnableSequence(chat_prompt | llm)
#             response = call_cohere_api(chat_chain, {"question": query.question})
#             return {
#                 "answer": response.content.strip(),
#                 "sources": []
#             }
#
#         # -----------------------
#         # OUTFIT SUGGESTION PATH
#         # -----------------------
#         if intent == "gợi ý set đồ" and parts:
#             product_suggestions = []
#             filtered_products = fetch_product_descriptions(gender=gender, max_price=max_price)
#             filtered_docs = [
#                 Document(
#                     page_content=f"{p['name']}: {p['description']}",
#                     metadata={
#                         "id": p["id"],
#                         "name": p["name"],
#                         "gender": p["gender"],
#                         "price": p["price"],
#                         "thumbnail": p["thumbnail"],
#                         "category": p["category"]
#                     }
#                 ) for p in filtered_products
#                 if (p["gender"] and (p["gender"].upper() == gender.upper() or p["gender"].upper() == UNISEX_LABEL))
#             ]
#             if not filtered_docs:
#                 return {
#                     "answer": format_answer_to_text(intent, parts, [], query.question),
#                     "sources": []
#                 }
#             filtered_db = FAISS.from_documents(filtered_docs, embedding_model)
#             retriever = filtered_db.as_retriever(search_kwargs={"k": 5})
#             for part in parts:
#                 if ':' not in part:
#                     product_suggestions.append((part, "Định dạng không hợp lệ, bỏ qua."))
#                     continue
#                 part_category, part_desc = part.split(':', 1)
#                 part_category = part_category.strip().lower()
#                 part_desc = part_desc.strip()
#                 if "áo" in part_category or "blazer" in part_category or "sơ mi" in part_category:
#                     category_key = "áo"
#                 elif "quần" in part_category or "pants" in part_category or "jeans" in part_category:
#                     category_key = "quần"
#                 elif "giày" in part_category or "sandal" in part_category or "dép" in part_category:
#                     category_key = "giày"
#                 else:
#                     product_suggestions.append((part, "Không xác định được danh mục."))
#                     continue
#                 products_for_part = fetch_product_descriptions(category_key, gender, max_price)
#                 if not products_for_part:
#                     product_suggestions.append((part, f"Không tìm thấy sản phẩm phù hợp cho danh mục {category_key}."))
#                     continue
#                 products_for_part = [p for p in products_for_part if p.get("gender") and (
#                             p["gender"].upper() == gender.upper() or p["gender"].upper() == UNISEX_LABEL)]
#                 if not products_for_part:
#                     product_suggestions.append((part, f"Không tìm thấy sản phẩm phù hợp với giới tính {gender}."))
#                     continue
#                 part_docs = [
#                     Document(
#                         page_content=f"{p['name']}: {p['description']}",
#                         metadata={
#                             "id": p["id"],
#                             "name": p["name"],
#                             "gender": p["gender"],
#                             "price": p["price"],
#                             "thumbnail": p["thumbnail"],
#                             "category": p["category"],
#                             "part": part
#                         }
#                     ) for p in products_for_part
#                 ]
#                 part_db = FAISS.from_documents(part_docs, embedding_model)
#                 qa_chain = RetrievalQA.from_chain_type(
#                     llm=llm,
#                     retriever=part_db.as_retriever(search_kwargs={"k": 3}),
#                     chain_type_kwargs={"prompt": qa_prompt_template},
#                     return_source_documents=True
#                 )
#                 try:
#                     retriever_results = part_db.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(part_desc)
#                     print(f"Retriever results for part '{part}': {[doc.page_content for doc in retriever_results]}")
#                 except Exception as e:
#                     print(f"Warning: retriever.get_relevant_documents failed: {e}")
#                 print(f"Invoking QA chain for part '{part}' with question: {part_desc}")
#                 qa_response = call_cohere_api(qa_chain, {"query": part_desc})
#                 print(f"QA response for part '{part}': {qa_response.get('result', '')[:200]}...")
#                 part_suggestion = qa_response.get("result", "Không tìm thấy sản phẩm phù hợp.")
#                 for doc in qa_response.get("source_documents", []):
#                     meta = doc.metadata
#                     sources_list.append({
#                         "id": meta.get("id"),
#                         "name": meta.get("name"),
#                         "price": meta.get("price"),
#                         "thumbnail": meta.get("thumbnail", ""),
#                         "description": doc.page_content,
#                         "gender": meta.get("gender"),
#                         "category": meta.get("category"),
#                         "part": meta.get("part")
#                     })
#                 product_suggestions.append((part, part_suggestion))
#             return {
#                 "answer": format_answer_to_text(intent, parts, product_suggestions, None, query.question),
#                 "sources": sources_list
#             }
#         # -----------------------
#         # SINGLE CATEGORY PATH
#         # -----------------------
#         elif intent == "gợi ý sản phẩm" and category:
#             filtered_products = fetch_product_descriptions(category, gender, max_price)
#             if not filtered_products:
#                 return {
#                     "answer": format_answer_to_text(intent, [], None, {"category": category,
#                                                                        "answer": f"Không tìm thấy {category} phù hợp với yêu cầu."},
#                                                     query.question),
#                     "sources": []
#                 }
#             filtered_docs = [
#                 Document(
#                     page_content=f"{p['name']}: {p['description']}",
#                     metadata={
#                         "id": p["id"],
#                         "name": p["name"],
#                         "gender": p["gender"],
#                         "price": p["price"],
#                         "thumbnail": p["thumbnail"],
#                         "category": p["category"]
#                     }
#                 ) for p in filtered_products
#                 if p.get("gender") and (p["gender"].upper() == gender.upper() or p["gender"].upper() == UNISEX_LABEL)
#             ]
#             if not filtered_docs:
#                 return {
#                     "answer": format_answer_to_text(intent, [], None, {"category": category,
#                                                                        "answer": f"Không tìm thấy {category} phù hợp với giới tính {gender}."},
#                                                     query.question),
#                     "sources": []
#                 }
#             filtered_db = FAISS.from_documents(filtered_docs, embedding_model)
#             retriever = filtered_db.as_retriever(search_kwargs={"k": 3})
#             context = "\n".join([
#                                     f"{doc.page_content} (Category: {doc.metadata['category']}, Gender: {doc.metadata['gender']}, Price: {doc.metadata['price']})"
#                                     for doc in filtered_docs])
#             print(f"Context for product suggestion: {context[:200]}...")
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=llm,
#                 retriever=retriever,
#                 chain_type_kwargs={"prompt": qa_prompt_template},
#                 return_source_documents=True
#             )
#             try:
#                 retriever_results = retriever.get_relevant_documents(query.question)
#                 print(f"Retriever results for query '{query.question}': {[doc.page_content for doc in retriever_results]}")
#             except Exception as e:
#                 print(f"Warning: retriever.get_relevant_documents failed: {e}")
#             print(f"Invoking QA chain with question: {query.question}")
#             qa_response = call_cohere_api(qa_chain, {"query": query.question})
#             print(f"QA response for query '{query.question}': {qa_response.get('result', '')[:200]}...")
#             analyzed_answer = qa_response.get("result", "Không tìm thấy sản phẩm phù hợp.")
#             for doc in qa_response.get("source_documents", []):
#                 meta = doc.metadata
#                 sources_list.append({
#                     "id": meta.get("id"),
#                     "name": meta.get("name"),
#                     "price": meta.get("price"),
#                     "thumbnail": meta.get("thumbnail", ""),
#                     "description": doc.page_content,
#                     "gender": meta.get("gender"),
#                     "category": meta.get("category")
#                 })
#             return {
#                 "answer": format_answer_to_text(intent, [], None, {"category": category, "answer": analyzed_answer},
#                                                 query.question),
#                 "sources": sources_list
#             }
#         # -----------------------
#         # FALLBACK PATH
#         # -----------------------
#         else:
#             return {
#                 "answer": "Tôi chưa hiểu rõ yêu cầu của bạn. Bạn có muốn trò chuyện bình thường hay cần gợi ý sản phẩm nào không?",
#                 "sources": []
#             }
#     except HTTPError as e:
#         if e.response.status_code == 429:
#             return {
#                 "answer": "Hệ thống hiện đang bận, vui lòng thử lại sau vài phút.",
#                 "sources": []
#             }
#         print(f"Error processing request: {e}")
#         return {
#             "answer": "Có lỗi xảy ra khi xử lý yêu cầu. Vui lòng thử lại.",
#             "sources": []
#         }
#     except Exception as e:
#         print(f"Error processing request: {e}")
#         return {
#             "answer": "Có lỗi xảy ra khi xử lý yêu cầu. Vui lòng thử lại.",
#             "sources": []
#         }


from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_cohere.chat_models import ChatCohere
import cohere
import os
from dotenv import load_dotenv
import base64
import re
import json
from starlette.middleware.cors import CORSMiddleware
import time
from functools import wraps
from requests.exceptions import HTTPError

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY not found in environment variables. Please check .env file.")

# --- DATABASE CONFIGURATION ---
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME", "fashion-shop-3"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "123456"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}
engine = create_engine(
    f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}")

# --- FETCH CATEGORIES FROM DATABASE ---
def fetch_categories():
    query = "SELECT id, name FROM public.categories"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            categories = {row.id: row.name.lower() for row in result}
            if not categories:
                raise ValueError("No categories found in database. Please ensure the 'categories' table is populated.")
            print(f"Fetched {len(categories)} categories from database: {categories}")
            return categories
    except Exception as e:
        print(f"Error fetching categories from database: {e}")
        raise ValueError("Failed to fetch categories from database. Application cannot start without valid categories.")

# Load categories at startup and validate
CATEGORY_MAPPING = fetch_categories()
REQUIRED_CATEGORIES = {"áo", "quần", "giày"}
if not all(cat in CATEGORY_MAPPING.values() for cat in REQUIRED_CATEGORIES):
    missing = REQUIRED_CATEGORIES - set(CATEGORY_MAPPING.values())
    raise ValueError(f"Missing required categories {missing} in CATEGORY_MAPPING: {CATEGORY_MAPPING}")

# --- VALID GENDERS ---
VALID_GENDERS = ['MALE', 'FEMALE']
UNISEX_LABEL = "UNISEX"

# --- RATE LIMITER DECORATOR ---
def rate_limit(max_per_minute):
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            time_to_wait = min_interval - elapsed
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

# --- FETCH PRODUCTS FROM POSTGRES WITH FILTERS ---
def fetch_product_descriptions(category=None, gender=None, max_price=None):
    query = "SELECT id, name, description, gender, price, category_id, thumbnail FROM public.products WHERE status = 't'"
    params = {}
    if category:
        category_id = next((k for k, v in CATEGORY_MAPPING.items() if v == category.lower()), None)
        if category_id is None:
            print(f"Error: Category '{category}' not found in CATEGORY_MAPPING: {CATEGORY_MAPPING}")
            return []
        query += " AND category_id = :category_id"
        params['category_id'] = category_id
    if gender:
        if gender.upper() in VALID_GENDERS or gender.upper() == UNISEX_LABEL:
            query += " AND gender = :gender"
            params['gender'] = gender.upper()
        else:
            print(f"Warning: Invalid gender '{gender}', skipping gender filter.")

    if max_price:
        query += " AND price <= :max_price"
        params['max_price'] = float(max_price)

    try:
        with engine.connect() as conn:
            print(f"Executing SQL: {query}, Parameters: {params}")
            result = conn.execute(text(query), params)
            products = []
            for row in result:
                thumb_url = "https://via.placeholder.com/120"
                if row.thumbnail:
                    if isinstance(row.thumbnail, (bytes, bytearray)):
                        b64_str = base64.b64encode(row.thumbnail).decode("utf-8")
                        thumb_url = f"data:image/jpeg;base64,{b64_str}"
                    elif isinstance(row.thumbnail, str):
                        thumb_url = row.thumbnail

                products.append({
                    "id": str(row.id),
                    "name": row.name,
                    "description": row.description,
                    "gender": row.gender,
                    "price": float(row.price),
                    "thumbnail": thumb_url,
                    "category": CATEGORY_MAPPING.get(row.category_id, "khác")
                })
            print(f"Fetched {len(products)} products for category: {category}, gender: {gender}, max_price: {max_price}")
            return products
    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return []

# --- PARSE USER QUERY USING COHERE LLM ---
llm = ChatCohere(cohere_api_key=cohere_api_key)

parse_prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
    Phân tích câu hỏi của người dùng bằng tiếng Việt một cách chính xác:
    - Nếu chứa từ như 'đồ', 'bộ đồ', 'set', 'set đồ', 'outfit' hoặc không chỉ định danh mục cụ thể (áo, quần, giày), thì intent là 'gợi ý set đồ' và phân tích thành 3 phần danh mục (áo, quần, giày) với mô tả ngắn gọn cho từng phần.
    - Nếu chỉ định chính xác 1 danh mục (ví dụ: áo, quần, giày, váy, phụ kiện), thì intent là 'gợi ý sản phẩm', category là danh mục đó, không lẫn lộn các danh mục khác.
    - Nếu câu hỏi không chứa từ khóa liên quan đến sản phẩm (như áo, quần, giày, set, đồ, giá, v.v.) và mang tính trò chuyện (ví dụ: 'chào', 'hôm nay thế nào', 'kể chuyện'), thì intent là 'trò chuyện'.
    - Trích xuất giới tính (nam -> 'MALE', nữ -> 'FEMALE', hoặc '' nếu không xác định).
    - Trích xuất giá tối đa (nếu có, chuyển về số, ví dụ: 2 triệu -> 2000000).
    - Trả về **chỉ** JSON hợp lệ, không sử dụng markdown wrapper (```json hoặc ```):
    {{
      "intent": "gợi ý set đồ" hoặc "gợi ý sản phẩm" hoặc "trò chuyện",
      "category": "danh mục nếu không phải set hoặc trò chuyện, hoặc null",
      "gender": "MALE" hoặc "FEMALE" hoặc "",
      "max_price": null hoặc số,
      "parts": [] nếu set đồ thì là list 3 phần mô tả, ví dụ ["Áo: mô tả", "Quần: mô tả", "Giày: mô tả"], nếu không phải set thì []
    }}
    Câu hỏi: {query}
    """
)

@rate_limit(max_per_minute=10)
def call_cohere_api(chain, input_data):
    return chain.invoke(input_data)

def parse_user_query(query):
    parse_chain = RunnableSequence(parse_prompt_template | llm)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = call_cohere_api(parse_chain, {"query": query})
            raw_output = result.content.strip()
            print(f"Raw LLM output (attempt {attempt + 1}): {raw_output}")
            raw_output = raw_output.encode('utf-8').decode('utf-8-sig')
            raw_output = re.sub(r'^```json\n|\n```$', '', raw_output).strip()
            print(f"Cleaned output (attempt {attempt + 1}): {raw_output}")
            if not raw_output or not (raw_output.startswith('{') and raw_output.endswith('}')):
                print(f"Error: LLM output is not valid JSON (attempt {attempt + 1})")
                continue
            parsed = json.loads(raw_output)
            print(f"Parsed JSON: {parsed}")
            gender = parsed.get("gender") or ""
            if gender and gender.upper() not in VALID_GENDERS:
                if gender.upper() == UNISEX_LABEL:
                    pass
                else:
                    print(f"Warning: Invalid gender '{gender}', defaulting to ''")
                    gender = ""
            parts = parsed.get("parts", [])[:3]  # Ensure only 3 parts
            if len(parsed.get("parts", [])) > 3:
                print(f"Warning: LLM returned {len(parsed.get('parts', []))} parts, truncating to 3.")
            return (
                parsed.get("intent"),
                parsed.get("category"),
                gender,
                parsed.get("max_price"),
                parts
            )
        except json.JSONDecodeError as e:
            print(f"JSON parse error in parse_user_query (attempt {attempt + 1}): {e}")
            print(f"Raw output causing error: {repr(raw_output)}")
            continue
        except Exception as e:
            print(f"Error parsing query (attempt {attempt + 1}): {e}")
            continue
    print("Error: Failed to parse JSON after all attempts")
    if "áo" in query.lower() and "nam" in query.lower():
        return "gợi ý sản phẩm", "áo", "MALE", None, []
    return None, None, None, None, []

# --- EMBEDDING WITH COHERE ---
co = cohere.Client(cohere_api_key)
embedding_model = CohereEmbeddings(
    cohere_api_key=cohere_api_key,
    model="embed-multilingual-v3.0"
)

# --- LOAD DATA AND CREATE INITIAL VECTOR STORE ---
products = fetch_product_descriptions()
if not products:
    print("No products found in database.")
docs = [Document(page_content=f"{p['name']}: {p['description']}", metadata={
    "id": p["id"],
    "name": p["name"],
    "gender": p["gender"],
    "price": p["price"],
    "thumbnail": p["thumbnail"],
    "category": p["category"]
}) for p in products]
db = FAISS.from_documents(docs, embedding_model)

# --- PROMPT FOR RETRIEVAL QA (PRODUCT MATCHING) ---
qa_prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    Dựa trên mô tả: {question}
    Hãy gợi ý sản phẩm phù hợp từ context, đảm bảo danh mục đúng (ví dụ: áo, quần, giày, váy, phụ kiện).
    Context: {context}
    Trả về mô tả sản phẩm phù hợp dưới dạng văn bản tự nhiên bằng tiếng Việt, không sử dụng JSON hoặc markdown. Ví dụ: "Áo sơ mi nữ H&M, chất liệu cotton pha lụa, màu trắng kem, phù hợp cho công sở."
    """
)

# --- HELPER FUNCTION TO FORMAT ANSWER FIELD AS USER-FRIENDLY TEXT ---
def format_answer_to_text(intent, parts, product_suggestions=None, single_category_answer=None, query=""):
    if intent == "gợi ý set đồ" and parts:
        header = "Dựa trên yêu cầu của bạn, đây là gợi ý set đồ:" if "set đồ" in query.lower() else "Dựa trên yêu cầu của bạn, đây là gợi ý:"
        response_text = f"{header}\n\n"
        response_text += "Set đồ gợi ý:\n"
        for part in parts:
            response_text += f"- {part}\n"
        response_text += "\n"
        if product_suggestions:
            response_text += "Chi tiết sản phẩm:\n"
            for part, suggestion in product_suggestions:
                response_text += f"- {part}\n"
                response_text += f"  {suggestion}\n"
        return response_text.strip()
    elif intent == "gợi ý sản phẩm" and single_category_answer:
        header = f"Dựa trên yêu cầu của bạn, đây là gợi ý sản phẩm cho {single_category_answer.get('category', '')}:"
        response_text = f"{header}\n\n"
        response_text += single_category_answer.get("answer", "Không tìm thấy sản phẩm phù hợp.")
        return response_text.strip()
    return "Không thể xử lý yêu cầu. Vui lòng thử lại."

router = APIRouter()

class Query(BaseModel):
    question: str

@router.post("/chat")
def chat(query: Query):
    try:
        intent, category, gender, max_price, parts = parse_user_query(query.question)
        print(f"Parsed intent: {intent}, category: {category}, gender: {gender}, max_price: {max_price}, parts: {parts}")
        sources_list = []

        # -----------------------
        # TRÒ CHUYỆN PATH
        # -----------------------
        if intent == "trò chuyện":
            # Gọi LLM trực tiếp để trả lời tự nhiên
            chat_prompt = PromptTemplate(
                input_variables=["question"],
                template="""
                Trả lời câu hỏi của người dùng bằng tiếng Việt một cách tự nhiên, thân thiện, như một cuộc trò chuyện thông thường. Không đề cập đến sản phẩm, gợi ý mua sắm, hoặc bất kỳ nội dung liên quan đến cửa hàng trừ khi được yêu cầu rõ ràng.
                Câu hỏi: {question}
                """
            )
            chat_chain = RunnableSequence(chat_prompt | llm)
            response = call_cohere_api(chat_chain, {"question": query.question})
            return {
                "answer": response.content.strip(),
                "sources": []
            }

        # -----------------------
        # OUTFIT SUGGESTION PATH
        # -----------------------
        if intent == "gợi ý set đồ" and parts:
            product_suggestions = []
            filtered_products = fetch_product_descriptions(gender=gender, max_price=max_price)
            filtered_docs = [
                Document(
                    page_content=f"{p['name']}: {p['description']}",
                    metadata={
                        "id": p["id"],
                        "name": p["name"],
                        "gender": p["gender"],
                        "price": p["price"],
                        "thumbnail": p["thumbnail"],
                        "category": p["category"]
                    }
                ) for p in filtered_products
                if (p["gender"] and (p["gender"].upper() == gender.upper() or p["gender"].upper() == UNISEX_LABEL))
            ]
            if not filtered_docs:
                return {
                    "answer": format_answer_to_text(intent, parts, [], query.question),
                    "sources": []
                }
            filtered_db = FAISS.from_documents(filtered_docs, embedding_model)
            retriever = filtered_db.as_retriever(search_kwargs={"k": 5})
            for part in parts:
                if ':' not in part:
                    product_suggestions.append((part, "Định dạng không hợp lệ, bỏ qua."))
                    continue
                part_category, part_desc = part.split(':', 1)
                part_category = part_category.strip().lower()
                part_desc = part_desc.strip()
                if "áo" in part_category or "blazer" in part_category or "sơ mi" in part_category:
                    category_key = "áo"
                elif "quần" in part_category or "pants" in part_category or "jeans" in part_category:
                    category_key = "quần"
                elif "giày" in part_category or "sandal" in part_category or "dép" in part_category:
                    category_key = "giày"
                else:
                    product_suggestions.append((part, "Không xác định được danh mục."))
                    continue
                products_for_part = fetch_product_descriptions(category_key, gender, max_price)
                if not products_for_part:
                    product_suggestions.append((part, f"Không tìm thấy sản phẩm phù hợp cho danh mục {category_key}."))
                    continue
                products_for_part = [p for p in products_for_part if p.get("gender") and (
                            p["gender"].upper() == gender.upper() or p["gender"].upper() == UNISEX_LABEL)]
                if not products_for_part:
                    product_suggestions.append((part, f"Không tìm thấy sản phẩm phù hợp với giới tính {gender}."))
                    continue
                part_docs = [
                    Document(
                        page_content=f"{p['name']}: {p['description']}",
                        metadata={
                            "id": p["id"],
                            "name": p["name"],
                            "gender": p["gender"],
                            "price": p["price"],
                            "thumbnail": p["thumbnail"],
                            "category": p["category"],
                            "part": part
                        }
                    ) for p in products_for_part
                ]
                part_db = FAISS.from_documents(part_docs, embedding_model)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=part_db.as_retriever(search_kwargs={"k": 3}),
                    chain_type_kwargs={"prompt": qa_prompt_template},
                    return_source_documents=True
                )
                try:
                    retriever_results = part_db.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(part_desc)
                    print(f"Retriever results for part '{part}': {[doc.page_content for doc in retriever_results]}")
                except Exception as e:
                    print(f"Warning: retriever.get_relevant_documents failed: {e}")
                print(f"Invoking QA chain for part '{part}' with question: {part_desc}")
                qa_response = call_cohere_api(qa_chain, {"query": part_desc})
                print(f"QA response for part '{part}': {qa_response.get('result', '')[:200]}...")
                part_suggestion = qa_response.get("result", "Không tìm thấy sản phẩm phù hợp.")
                for doc in qa_response.get("source_documents", []):
                    meta = doc.metadata
                    sources_list.append({
                        "id": meta.get("id"),
                        "name": meta.get("name"),
                        "price": meta.get("price"),
                        "thumbnail": meta.get("thumbnail", ""),
                        "description": doc.page_content,
                        "gender": meta.get("gender"),
                        "category": meta.get("category"),
                        "part": meta.get("part")
                    })
                product_suggestions.append((part, part_suggestion))
            return {
                "answer": format_answer_to_text(intent, parts, product_suggestions, None, query.question),
                "sources": sources_list
            }
        # -----------------------
        # SINGLE CATEGORY PATH
        # -----------------------
        elif intent == "gợi ý sản phẩm" and category:
            filtered_products = fetch_product_descriptions(category, gender, max_price)
            if not filtered_products:
                return {
                    "answer": format_answer_to_text(intent, [], None, {"category": category,
                                                                       "answer": f"Không tìm thấy {category} phù hợp với yêu cầu."},
                                                    query.question),
                    "sources": []
                }
            filtered_docs = [
                Document(
                    page_content=f"{p['name']}: {p['description']}",
                    metadata={
                        "id": p["id"],
                        "name": p["name"],
                        "gender": p["gender"],
                        "price": p["price"],
                        "thumbnail": p["thumbnail"],
                        "category": p["category"]
                    }
                ) for p in filtered_products
                if p.get("gender") and (p["gender"].upper() == gender.upper() or p["gender"].upper() == UNISEX_LABEL)
            ]
            if not filtered_docs:
                return {
                    "answer": format_answer_to_text(intent, [], None, {"category": category,
                                                                       "answer": f"Không tìm thấy {category} phù hợp với giới tính {gender}."},
                                                    query.question),
                    "sources": []
                }
            filtered_db = FAISS.from_documents(filtered_docs, embedding_model)
            retriever = filtered_db.as_retriever(search_kwargs={"k": 3})
            context = "\n".join([
                                    f"{doc.page_content} (Category: {doc.metadata['category']}, Gender: {doc.metadata['gender']}, Price: {doc.metadata['price']})"
                                    for doc in filtered_docs])
            print(f"Context for product suggestion: {context[:200]}...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": qa_prompt_template},
                return_source_documents=True
            )
            try:
                retriever_results = retriever.get_relevant_documents(query.question)
                print(f"Retriever results for query '{query.question}': {[doc.page_content for doc in retriever_results]}")
            except Exception as e:
                print(f"Warning: retriever.get_relevant_documents failed: {e}")
            print(f"Invoking QA chain with question: {query.question}")
            qa_response = call_cohere_api(qa_chain, {"query": query.question})
            print(f"QA response for query '{query.question}': {qa_response.get('result', '')[:200]}...")
            analyzed_answer = qa_response.get("result", "Không tìm thấy sản phẩm phù hợp.")
            for doc in qa_response.get("source_documents", []):
                meta = doc.metadata
                sources_list.append({
                    "id": meta.get("id"),
                    "name": meta.get("name"),
                    "price": meta.get("price"),
                    "thumbnail": meta.get("thumbnail", ""),
                    "description": doc.page_content,
                    "gender": meta.get("gender"),
                    "category": meta.get("category")
                })
            return {
                "answer": format_answer_to_text(intent, [], None, {"category": category, "answer": analyzed_answer},
                                                query.question),
                "sources": sources_list
            }
        # -----------------------
        # FALLBACK PATH
        # -----------------------
        else:
            return {
                "answer": "Tôi chưa hiểu rõ yêu cầu của bạn. Bạn có muốn trò chuyện bình thường hay cần gợi ý sản phẩm nào không?",
                "sources": []
            }
    except HTTPError as e:
        if e.response.status_code == 429:
            return {
                "answer": "Hệ thống hiện đang bận, vui lòng thử lại sau vài phút.",
                "sources": []
            }
        print(f"Error processing request: {e}")
        return {
            "answer": "Có lỗi xảy ra khi xử lý yêu cầu. Vui lòng thử lại.",
            "sources": []
        }
    except Exception as e:
        print(f"Error processing request: {e}")
        return {
            "answer": "Có lỗi xảy ra khi xử lý yêu cầu. Vui lòng thử lại.",
            "sources": []
        }