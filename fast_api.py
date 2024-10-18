from fastapi import FastAPI
from pydantic import BaseModel
from elasticsearch import Elasticsearch
import os
from openai import OpenAI
import time 
import cohere



app = FastAPI()
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
elastic_api_key = os.getenv("ELASTIC_API_KEY")

class Query(BaseModel):
    question: str


# Initialize OpenAI model
model = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

# Connect to Elasticsearch
client = Elasticsearch(
    "http://localhost:9200/",
    api_key=elastic_api_key,
)


index_name = "text_log"

# Function to format Elasticsearch response
def pretty_response(response):
    pretty_text = ""
    for hit in response["hits"]["hits"]:
        id = hit["_id"]
        score = hit["_score"]
        method = hit["_source"]["method"]
        user_agent = hit["_source"]["user_agent"]
        url = hit["_source"]["url"]
        ip = hit["_source"]["ip"]
        des = hit["_source"]["des"]
        date = hit["_source"]["date"]
        
        pretty_output = (
            f"\nID: {id}\n"
            f"User Agent: {user_agent}\n"
            f"URL: {url}\n"
            f"IP: {ip}\n"
            f"Method: {method}\n"
            f"Date : {date}\n"
            f"Relevance Score: {score}\n"
            f"Description: {des}\n"
        )


        print(pretty_output)
        pretty_text += pretty_output
    return pretty_text

#json hit to rerank
def json_hit(response):
    docs = []
    for hit in response['hits']['hits']:
        doc = {
            "user_agent": str(hit["_source"]["user_agent"]),
            "url": str(hit["_source"]["url"]),
            "ip": str(hit["_source"]["ip"]),
            "method": str(hit["_source"]["method"]),
            "date": str(hit["_source"]["date"]),
            "des" : str(hit["_source"]["des"])
        }
        docs.append(doc)
    return docs


#question = "Find logs where users viewed products from a Windows platform using Edge after july 2023."

@app.post("/query")
async def ragModel(query: Query):
    
    question = query.question
    # Vectorize the question
    question_embedding = model.embeddings.create(input=question, model=EMBEDDING_MODEL)

    filter_start = time.time()
    response_date = model.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {
                "role": "system",
                "content": (
                    "Based on the provided query, generate a date range suitable for Elasticsearch filtering in the format 'gte' and 'lte'. "
                    "If 'gte' is not specified in the query, set it to a reasonable date in the back (e.g., '2015-01-01T00:00:00.000000Z'). "
                    "If 'lte' is not mentioned, return '2025-01-01T00:00:00.000000Z', if you find a date return it using the format 'YYYY-MM-DDTHH:MM:SS.SSSSSSZ'. "
                    "In this case, since the query specifies a cutoff date, use that as 'lte'.if it ask for after use gte, or you can use both in case such as between "
                    "Additionally, simplify the query to capture key terms relevant to reranking (e.g., method, user action, device type | OS, or browser). "
                    "For example, if the input is 'user updated his password using faceID from Safari', the simplified rerank query would be 'update password safari'. "
                    "Return the date range in the format 'gte lte' separated by a space, followed by the simplified rerank query. "
                    "Do not generate anything else; the final result should be 'gte lte rerquery' with valid dates without '."
                )
            },
            {"role": "user", "content": question}
        ]
    )

    string_data = response_date.choices[0].message.content
    print(string_data)

    #Step 2: Split the response by space to get gte and lte
    gte_str, lte_str, rerank_query = string_data.split(" ", 2)


    print("gte:", gte_str)
    print("lte:", lte_str)
    print("rerank:", rerank_query)

    filter_end = time.time()

    search_start = time.time()
    # Perform knn search on Elasticsearch index on text
    size = 10
    response = client.search(
        index=index_name,
        query={
            "range": {
                "date": { #filter by date
                    "gte": gte_str,
                    "lte": lte_str,
                    "boost": 2
                }
            }
        },
        knn={
            "field": "vector_des",
            "query_vector": question_embedding.data[0].embedding,
            "k": size,
            "num_candidates": 500,
        },
        size=size
    )

    search_end = time.time()

    pretty_text = pretty_response(response)


    co_start = time.time()
    co = cohere.ClientV2(cohere_api_key)


    docs = json_hit(response)
    rank_fields = ["user_agent", "url", "method", "body"]
    top = 3

    rerank_response = co.rerank(

        model="rerank-english-v3.0",

        query=rerank_query,

        documents=docs,

        top_n=top,

        rank_fields=rank_fields
    )

    formatted_reranked_text = ""

    
    for result in rerank_response.results:

        doc = docs[result.index]

        formatted_output = (
            f"\nRelevance Score: {result.relevance_score}\n"
            f"User Agent: {doc['user_agent']}\n"
            f"URL: {doc['url']}\n"
            f"IP: {doc['ip']}\n"
            f"Method: {doc['method']}\n"
            f"Date: {doc['date']}\n"
            f"Description: {doc['des']}\n"
            "-----\n"
        )

        # Append the formatted output to the main string
        formatted_reranked_text += formatted_output

        print(f"Relevance Score: {result.relevance_score}")
        print(f"User Agent: {doc['user_agent']}")
        print(f"URL: {doc['url']}")
        print(f"IP: {doc['ip']}")
        print(f"Method: {doc['method']}")
        print(f"Date: {doc['date']}")
        print(f"Description: {doc['des']}")
        print("-----")


    co_end = time.time()


    gen_start = time.time()

    # Generate chat 
    completion = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
             {"role": "system", "content": 
                "You will be provided with web log entries that contain descriptions of user actions, such as requests made to different URLs using various HTTP methods" 
                "(e.g., GET, POST, PUT). The logs are retrieved based on a user's query, and your task is to analyze these logs and generate a suitable response.\n"
                "If the retrieved logs indicate actions that suggest an update (like a PUT request to /settings or /login), it could imply actions such as password updates," 
                "even if not explicitly mentioned. Infer the most logical conclusion and provide the information accordingly.\n"
                "For example, if the logs show a PUT request on /settings with a user agent mentioning Safari, you may deduce that this action is related to updating user account details.try to give short answers"
            },
            {"role": "user", "content": pretty_text},
            {"role": "user", "content": formatted_reranked_text},

        ],
    )
    gen_end = time.time()

    message = completion.choices[0].message.content
    print(message)
    
    print("\n")
    print(f"filter Time: {filter_end - filter_start} seconds")
    print(f"Search Time: {search_end - search_start} seconds")
    print(f"Reranking Time: {co_end - co_start} seconds")
    print(f"generation Time: {gen_end - gen_start} seconds")

    return {
        "filtered_results": pretty_text,
        "reranked_results": formatted_reranked_text,
        "final_message": message
    }




    
