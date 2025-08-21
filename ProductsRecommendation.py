from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from pyvi import ViTokenizer
from gensim.models import Word2Vec
import uvicorn

router = APIRouter()

# Định nghĩa Pydantic models cho định dạng JSON
class Brand(BaseModel):
    id: str
    name: str

class Category(BaseModel):
    id: str
    code: str
    name: str

class Product(BaseModel):
    id: str
    created_at: str
    created_by: str
    updated_at: str
    updated_by: str
    name: str
    description: str
    price: float
    gender: str
    sold: int
    rating: float
    status: bool
    thumbnail: str
    slider: str | None
    category_id: str
    brand_id: str
    brand: Brand
    category: Category
    similarity: float | None = None  # Chỉ dùng cho content-based recommendations

class ResponseModel(BaseModel):
    status: str
    statusCode: int
    message: str
    data: List[Product]

def connect_to_database():
    engine = create_engine('postgresql+psycopg2://postgres:123456@localhost:5432/fashion-shop-3')
    return engine

def load_data():
    engine = connect_to_database()
    interactions_query = "SELECT user_id, product_id, interaction_type FROM user_interactions"
    product_query = """
        SELECT 
            p.id, p.created_at, p.created_by, p.updated_at, p.updated_by, 
            p.name, p.description, p.price, p.gender, p.sold, p.rating, 
            p.status, p.thumbnail, p.slider, p.category_id, p.brand_id,
            b.id as brand_id_ref, b.name as brand_name,
            c.id as category_id_ref, c.code as category_code, c.name as category_name
        FROM products p
        LEFT JOIN brands b ON p.brand_id = b.id
        LEFT JOIN categories c ON p.category_id = c.id
    """

    try:
        user_product_df = pd.read_sql(interactions_query, engine).drop_duplicates()
        user_product_df = user_product_df.groupby(['user_id', 'product_id'], as_index=False).agg({'interaction_type': 'sum'})
        user_product_matrix = user_product_df.pivot_table(index='user_id', columns='product_id', values='interaction_type', fill_value=0)
        products_df = pd.read_sql(product_query, engine)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi cơ sở dữ liệu: {str(e)}")

    return user_product_matrix, products_df

def preprocess_text(text, stopwords, split=False):
    text = ViTokenizer.tokenize(text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    tokens = [word for word in text.split() if word not in stopwords]
    return tokens if split else ' '.join(tokens)

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(f.read().splitlines())

stopwords_file_path = 'vietnamese-stopwords.txt'
stopwords = load_stopwords(stopwords_file_path)

# Tải dữ liệu và huấn luyện mô hình lần đầu
user_product_matrix, products_df = load_data()
processed_descriptions = [preprocess_text(desc or "", stopwords, split=True) for desc in products_df['description']]
model = Word2Vec(sentences=processed_descriptions, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

def get_product_vector(text, model):
    tokens = preprocess_text(text or "", stopwords, split=True)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        print(f"Cảnh báo: Mô tả '{text}' không tạo được vector (rỗng hoặc chỉ chứa stopwords).")
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

product_vectors = []
for desc in products_df['description']:
    vector = get_product_vector(desc, model)
    if np.all(vector == 0):
        print(f"Cảnh báo: Vector sản phẩm cho mô tả '{desc}' là vector 0.")
    product_vectors.append(vector)

try:
    similarity_matrix = cosine_similarity(product_vectors)
    product_ids = products_df['id'].values
except Exception as e:
    print(f"Lỗi khi tính cosine similarity: {str(e)}")
    similarity_matrix = np.zeros((len(products_df), len(products_df)))

def recommend_products(user_id: int) -> Dict[str, Any]:
    def format_product(product_id: str) -> Dict[str, Any]:
        product_row = products_df[products_df['id'] == product_id].iloc[0]
        return {
            "id": str(product_row['id']),
            "created_at": product_row['created_at'].isoformat() + "Z" if pd.notnull(product_row['created_at']) else "2025-01-01T00:00:00.000Z",
            "created_by": str(product_row['created_by']) if pd.notnull(product_row['created_by']) else "system",
            "updated_at": product_row['updated_at'].isoformat() + "Z" if pd.notnull(product_row['updated_at']) else "2025-01-01T00:00:00.000Z",
            "updated_by": str(product_row['updated_by']) if pd.notnull(product_row['updated_by']) else "system",
            "name": str(product_row['name']) if pd.notnull(product_row['name']) else "",
            "description": str(product_row['description']) if pd.notnull(product_row['description']) else "",
            "price": float(product_row['price']) if pd.notnull(product_row['price']) else 0.0,
            "gender": str(product_row['gender']) if pd.notnull(product_row['gender']) else "UNISEX",
            "sold": int(product_row['sold']) if pd.notnull(product_row['sold']) else 0,
            "rating": float(product_row['rating']) if pd.notnull(product_row['rating']) else 0.0,
            "status": bool(product_row['status']) if pd.notnull(product_row['status']) else True,
            "thumbnail": str(product_row['thumbnail']) if pd.notnull(product_row['thumbnail']) else "",
            "slider": str(product_row['slider']) if pd.notnull(product_row['slider']) else None,
            "category_id": str(product_row['category_id']) if pd.notnull(product_row['category_id']) else "",
            "brand_id": str(product_row['brand_id']) if pd.notnull(product_row['brand_id']) else "",
            "brand": {
                "id": str(product_row['brand_id_ref']) if pd.notnull(product_row['brand_id_ref']) else "",
                "name": str(product_row['brand_name']) if pd.notnull(product_row['brand_name']) else ""
            },
            "category": {
                "id": str(product_row['category_id_ref']) if pd.notnull(product_row['category_id_ref']) else "",
                "code": str(product_row['category_code']) if pd.notnull(product_row['category_code']) else "",
                "name": str(product_row['category_name']) if pd.notnull(product_row['category_name']) else ""
            }
        }

    try:
        if user_id == -1:
            popular_products = user_product_matrix.sum().sort_values(ascending=False).index[:8].tolist()
            return {
                "status": "success",
                "statusCode": 200,
                "message": "Product retrieved successfully",
                "data": [format_product(pid) for pid in popular_products]
            }

        knn = NearestNeighbors(n_neighbors=min(5, len(user_product_matrix)), algorithm='auto', metric='cosine')
        knn.fit(user_product_matrix)
        user_idx = user_product_matrix.index.get_loc(user_id) if user_id in user_product_matrix.index else None

        if user_idx is not None:
            user_data = user_product_matrix.iloc[[user_idx]]  # Giữ dạng DataFrame để tránh cảnh báo
            distances, indices = knn.kneighbors(user_data)
            similar_users = indices[0]
            similarity_weights = 1 / (distances[0] + 1e-5)

            weighted_scores = {}
            for i, user in enumerate(similar_users):
                user_products = user_product_matrix.iloc[user]
                for product in user_products[user_products > 0].index:
                    weighted_scores[product] = weighted_scores.get(product, 0) + similarity_weights[i]

            unread_products = user_product_matrix.iloc[user_idx] == 0
            recommended_products = {
                product: score
                for product, score in weighted_scores.items()
                if product in user_product_matrix.columns and unread_products[product]
            }

            sorted_products = sorted(recommended_products, key=recommended_products.get, reverse=True)

            if len(sorted_products) < 8:
                popular_products = user_product_matrix.sum().sort_values(ascending=False).index.tolist()
                unread_popular_products = [p for p in popular_products if p not in sorted_products and unread_products.get(p, True)]
                sorted_products.extend(unread_popular_products[:8 - len(sorted_products)])

            return {
                "status": "success",
                "statusCode": 200,
                "message": "Product retrieved successfully",
                "data": [format_product(pid) for pid in sorted_products[:8]]
            }

        popular_products = user_product_matrix.sum().sort_values(ascending=False).index[:8].tolist()
        return {
            "status": "success",
            "statusCode": 200,
            "message": "Product retrieved successfully",
            "data": [format_product(pid) for pid in popular_products]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo gợi ý: {str(e)}")

def content_based_recommendations(product_id: str) -> Dict[str, Any]:
    if product_id not in products_df['id'].values:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy sản phẩm với ID '{product_id}'.")

    product_index = products_df.index[products_df['id'] == product_id].tolist()[0]
    try:
        cosine_similarities = cosine_similarity([product_vectors[product_index]], product_vectors).flatten()
        if np.all(cosine_similarities == 0):
            print(f"Cảnh báo: Độ tương đồng cho sản phẩm '{product_id}' là 0. Kiểm tra mô tả sản phẩm hoặc model Word2Vec.")
        similar_indices = cosine_similarities.argsort()[-7:-1][::-1]
    except Exception as e:
        print(f"Lỗi khi tính độ tương đồng cho sản phẩm '{product_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tính độ tương đồng: {str(e)}")

    data = [
        {
            "id": str(products_df.iloc[i]['id']),
            "created_at": products_df.iloc[i]['created_at'].isoformat() + "Z" if pd.notnull(products_df.iloc[i]['created_at']) else "2025-01-01T00:00:00.000Z",
            "created_by": str(products_df.iloc[i]['created_by']) if pd.notnull(products_df.iloc[i]['created_by']) else "system",
            "updated_at": products_df.iloc[i]['updated_at'].isoformat() + "Z" if pd.notnull(products_df.iloc[i]['updated_at']) else "2025-01-01T00:00:00.000Z",
            "updated_by": str(products_df.iloc[i]['updated_by']) if pd.notnull(products_df.iloc[i]['updated_by']) else "system",
            "name": str(products_df.iloc[i]['name']) if pd.notnull(products_df.iloc[i]['name']) else "",
            "description": str(products_df.iloc[i]['description']) if pd.notnull(products_df.iloc[i]['description']) else "",
            "price": float(products_df.iloc[i]['price']) if pd.notnull(products_df.iloc[i]['price']) else 0.0,
            "gender": str(products_df.iloc[i]['gender']) if pd.notnull(products_df.iloc[i]['gender']) else "UNISEX",
            "sold": int(products_df.iloc[i]['sold']) if pd.notnull(products_df.iloc[i]['sold']) else 0,
            "rating": float(products_df.iloc[i]['rating']) if pd.notnull(products_df.iloc[i]['rating']) else 0.0,
            "status": bool(products_df.iloc[i]['status']) if pd.notnull(products_df.iloc[i]['status']) else True,
            "thumbnail": str(products_df.iloc[i]['thumbnail']) if pd.notnull(products_df.iloc[i]['thumbnail']) else "",
            "slider": str(products_df.iloc[i]['slider']) if pd.notnull(products_df.iloc[i]['slider']) else None,
            "category_id": str(products_df.iloc[i]['category_id']) if pd.notnull(products_df.iloc[i]['category_id']) else "",
            "brand_id": str(products_df.iloc[i]['brand_id']) if pd.notnull(products_df.iloc[i]['brand_id']) else "",
            "brand": {
                "id": str(products_df.iloc[i]['brand_id_ref']) if pd.notnull(products_df.iloc[i]['brand_id_ref']) else "",
                "name": str(products_df.iloc[i]['brand_name']) if pd.notnull(products_df.iloc[i]['brand_name']) else ""
            },
            "category": {
                "id": str(products_df.iloc[i]['category_id_ref']) if pd.notnull(products_df.iloc[i]['category_id_ref']) else "",
                "code": str(products_df.iloc[i]['category_code']) if pd.notnull(products_df.iloc[i]['category_code']) else "",
                "name": str(products_df.iloc[i]['category_name']) if pd.notnull(products_df.iloc[i]['category_name']) else ""
            },
            "similarity": float(cosine_similarities[i]) if not np.isnan(cosine_similarities[i]) else None
        }
        for i in similar_indices
    ]

    return {
        "status": "success",
        "statusCode": 200,
        "message": "Product retrieved successfully",
        "data": data
    }

@router.post("/refresh_data/")
def refresh_data():
    global user_product_matrix, products_df, product_vectors, similarity_matrix
    try:
        user_product_matrix, products_df = load_data()
        processed_descriptions = [preprocess_text(desc or "", stopwords, split=True) for desc in products_df['description']]
        model = Word2Vec(sentences=processed_descriptions, vector_size=100, window=5, min_count=1, workers=4)
        model.save("word2vec.model")
        product_vectors = []
        for desc in products_df['description']:
            vector = get_product_vector(desc, model)
            if np.all(vector == 0):
                print(f"Cảnh báo: Vector sản phẩm cho mô tả '{desc}' là vector 0.")
            product_vectors.append(vector)
        similarity_matrix = cosine_similarity(product_vectors)
        return {
            "status": "success",
            "statusCode": 200,
            "message": "Data has been refreshed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi làm mới dữ liệu: {str(e)}")

@router.get("/recommendations/{user_id}", response_model=ResponseModel)
def get_recommendations(user_id: int):
    return recommend_products(user_id)

@router.get("/recommendations_by_product/{product_id}", response_model=ResponseModel)
def get_recommendations_by_product(product_id: str):
    return content_based_recommendations(product_id)


