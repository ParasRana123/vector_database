from fastapi import FastAPI
from contextlib import asynccontextmanager
import fasttext
import custom_vectordb
import os
import Levenshtein

ft_model=None
db=None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global ft_model
    global db
    
    print("Loading FastText model...")
    ft_model=fasttext.load_model("cc.en.300.bin")
    
    # Initialize Engine with strict 300 dimensions implicitly assumed in code structure, mapped to COSINE
    db=custom_vectordb.Engine(custom_vectordb.MetricType.COSINE)
    
    if os.path.exists("snapshot.bin"):
        print("Found snapshot.bin! Loading database from disk...")
        db.load_from_file("snapshot.bin")
    else:
        print("snapshot.bin not found. Building database directly from FastText binary model...")
        words = ft_model.get_words()
        db.reserve(len(words) + 1000)
        idx=0
        for word in words:
            vec=ft_model.get_word_vector(word).tolist()
            db.insert(str(idx),vec,word)
            idx+=1
            if idx%1000==0:
                print(f"Inserted {idx} words...")
            if idx>=500000:
                break
        print("Saving snapshot.bin to disk...")
        db.save_to_file("snapshot.bin")
    
    print("Database ready!")
    yield

from fastapi.middleware.cors import CORSMiddleware

app=FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)


def sim(v1,v2):
    return Levenshtein.ratio(v1,v2)

@app.get("/search")
def search(word:str,k:int, metric:str="cosine"):
    search_k= k+10
    vec=ft_model.get_word_vector(word).tolist()

    results=db.search(vec,search_k)

    count=0;

    final_results=[]
    for i in results:

        if(sim(i.metadata,word)<0.7):
            final_results.append(i)
            count+=1

        if(count==k):
            break

    results=final_results

    for i in final_results:
        print(i.metadata,sim(i.metadata,word))

    return {
        "query_vector": [round(v, 3) for v in vec],
        "results": [
            {
                "word": res.metadata,
                "distance": round(float(res.distance), 4),
                "vector": [round(v, 3) for v in ft_model.get_word_vector(res.metadata).tolist()]
    
            } for res in results
        ]
    }